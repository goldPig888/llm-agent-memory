"""One-worker 100-episode Glimmer baseline; preserve trial history in one viewer."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

PROJECT=Path(__file__).resolve().parents[1]
MODEL_ROOT=Path(os.environ.get('MUSE_ROOT',
    '/metadisk/may/muse-glimmer' if Path('/metadisk/may/muse-glimmer').exists()
    else str(Path.home()/'.local/share/muse-glimmer'))).expanduser().resolve()


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:
        return False


def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)); temp.replace(path)


def model_action_error(folder,results):
    """Recognize invalid rotation commands; preserve their original error logs."""
    if len(results)!=1 or results[0].get('status')!='error':
        return False
    for path in (folder/'episode_00/inspect').glob('*.json'):
        error=json.loads(path.read_text()).get('error') or ''
        if error.startswith('EmbodimentFault: LIBERO worker: ValueError: Degenerate rot6d'):
            return True
    return False


def save(root,manifest):
    write_json(root/'campaign.json',manifest)
    combined=dict(name='Muse Glimmer trials and baseline',runs=[])
    for source in [MODEL_ROOT/'results',MODEL_ROOT/'harness_xhigh_results',root]:
        path=source/'campaign.json'
        if not path.exists(): continue
        for original in json.loads(path.read_text()).get('runs',[]):
            row=copy.deepcopy(original)
            row['directory']=str((source/row['directory']).relative_to(MODEL_ROOT))
            row.setdefault('trial_label',source.name+' / '+original['directory'])
            combined['runs'].append(row)
    write_json(MODEL_ROOT/'campaign.json',combined)


def ensure_context(server_pid,root):
    with httpx.Client(trust_env=False,timeout=10) as client:
        props=client.get('http://127.0.0.1:8081/props').json()
    if props['default_generation_settings']['n_ctx']>=131072:
        return None
    cmdline=Path(f'/proc/{server_pid}/cmdline').read_bytes()
    if b'llama-server' not in cmdline or b'muse-glimmer' not in cmdline:
        raise RuntimeError('Model server PID identity changed')
    os.kill(server_pid,signal.SIGTERM)
    deadline=time.monotonic()+120
    while alive(server_pid):
        if time.monotonic()>deadline: raise RuntimeError('Model server did not stop')
        time.sleep(1)
    log=(root/'model_server.log').open('a')
    command=[str(MODEL_ROOT/'llama.cpp/build-cuda128/bin/llama-server'),
        '-m',str(MODEL_ROOT/'models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf'),
        '--mmproj',str(MODEL_ROOT/'models/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf'),
        '--alias','muse-glimmer','--host','127.0.0.1','--port','8081',
        '--n-gpu-layers','99','--split-mode','layer','--ctx-size','131072',
        '--parallel','1','--jinja','--flash-attn','on',
        '--cache-type-k','q8_0','--cache-type-v','q8_0',
        '--chat-template-kwargs','{"reasoning_strength":"xhigh"}']
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        if proc.poll() is not None: raise RuntimeError('Model server failed; inspect model_server.log')
        try:
            with httpx.Client(trust_env=False,timeout=5) as client:
                response=client.get('http://127.0.0.1:8081/health')
                if response.status_code==200:
                    props=client.get('http://127.0.0.1:8081/props').json()
                    if props['default_generation_settings']['n_ctx']<131072:
                        raise RuntimeError('Model context was reduced; baseline not started')
                    return proc
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError('Model did not become ready')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--after-pid',required=True,type=int)
    parser.add_argument('--server-pid',required=True,type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=args.resume)
    with (MODEL_ROOT/'.baseline_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(name='Muse Glimmer original-policy baseline xhigh / 16384',
            status='waiting_for_harness_trial',worker_count=1,after_pid=args.after_pid,
            model='muse-glimmer',effort='xhigh',max_output_tokens=16384,runs=[])
        for task in range(10):
            for state in range(10):
                manifest['runs'].append(dict(directory=f'episodes/task{task:02d}_state{state:02d}/rollout',
                    trial_label='Baseline xhigh 16384',model='muse-glimmer',effort='xhigh',
                    task_id=task,initial_state_ids=[state],status='pending',
                    max_calls=80,max_steps=900,max_output_tokens=16384,image_horizon=2))
        if args.resume:
            manifest=json.loads((root/'campaign.json').read_text())
            if manifest.get('status')!='stopped_for_inspection' or any(r['status']=='running' for r in manifest['runs']):
                raise RuntimeError('Resume requires a stopped campaign without running episodes')
            for row in manifest['runs']:
                summary=root/row['directory']/'summary.json'
                if summary.exists() and model_action_error(summary.parent,json.loads(summary.read_text()).get('results',[])):
                    row['termination_category']='model_action_error'
            manifest['after_pid']=args.after_pid
        save(root,manifest)
        print('Queued 100 baseline episodes; waiting for current harness trial.',flush=True)
        try:
            while alive(args.after_pid):
                save(root,manifest)
                time.sleep(15)
            manifest['status']='loading_full_context_model'; save(root,manifest)
            print('Harness trial finished; preparing full-context model.',flush=True)
            ensure_context(args.server_pid,root)
            manifest['status']='running'; save(root,manifest)
            for row in manifest['runs']:
                if row['status']!='pending':
                    continue
                row['status']='running'; save(root,manifest)
                state=row['initial_state_ids'][0]
                output=(root/row['directory']).parent
                command=[sys.executable,'-u',str(PROJECT/'scripts/run_local.py'),
                    '--output',str(output),'--run-name','rollout','--tasks',str(row['task_id']),
                    '--inits',str(state),'--effort','xhigh','--max-calls','80','--max-steps','900',
                    '--image-horizon','2','--max-output-tokens','16384']
                with (root/f'task{row["task_id"]:02d}_state{state:02d}.log').open('a') as log:
                    proc=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                    row['pid']=proc.pid
                    print(f'Started task {row["task_id"]} state {state}, PID {proc.pid}',flush=True)
                    while proc.poll() is None:
                        save(root,manifest); time.sleep(15)
                summary_path=root/row['directory']/'summary.json'
                summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
                results=summary.get('results',[])
                row.update(status='finished' if proc.returncode==0 and len(results)==1 else 'error',
                           exit_code=proc.returncode)
                action_error=model_action_error(root/row['directory'],results)
                if action_error:
                    row['termination_category']='model_action_error'
                    print('Invalid model rotation recorded; continuing to next episode.',flush=True)
                if row['status']=='error' or (any(r.get('status')=='error' for r in results) and not action_error):
                    manifest['status']='stopped_for_inspection'; save(root,manifest)
                    print('Infrastructure error; stopped before next episode.',flush=True)
                    return
                save(root,manifest)
            manifest['status']='finished'
        except BaseException:
            manifest['status']='stopped_for_inspection'
            raise
        finally:
            save(root,manifest)


if __name__=='__main__': main()

"""One-shot baseline watchdog, run every minute by a user systemd timer."""
import argparse
from datetime import datetime, timezone
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
PYTHON=sys.executable
STALE_SECONDS=2700


def atomic(path,data):
    temp=path.with_suffix('.watch.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def process_matches(pid,name,output=None):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        parts=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
        return state not in ('Z','X') and any(Path(v).name==name for v in parts) and (output is None or str(output) in parts)
    except (FileNotFoundError,PermissionError,UnicodeError,ProcessLookupError):
        return False


def processes(name,output=None):
    return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()
            and process_matches(int(p.name),name,output)]


def stop_tree(pid):
    """Stop only descendants of an identified rollout/controller process."""
    try:
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        for child in children: stop_tree(int(child))
        os.kill(pid,signal.SIGTERM)
    except (FileNotFoundError,ProcessLookupError):
        pass


def progress_time(folder):
    paths=list((folder/'api').glob('*request.json'))+list((folder/'api').glob('*response.json'))
    return max((p.stat().st_mtime for p in paths if p.exists()),default=0)


def reconcile(root,manifest):
    """Preserve final results; archive only unfinished attempts before retry."""
    for row in manifest['runs']:
        if row['status']!='running': continue
        folder=root/row['directory']
        summary=folder/'summary.json'
        results=json.loads(summary.read_text()).get('results',[]) if summary.exists() else []
        if results:
            row['status']='finished'
            row['recovery_note']='Recovered saved result after controller exit'
            continue
        if folder.exists():
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive=folder.with_name('rollout_interrupted_'+stamp)
            folder.rename(archive)
            row.setdefault('attempts',[]).append(dict(directory=str(archive.relative_to(root)),
                status='interrupted',reason='Watchdog recovered missing worker or stalled request'))
            child=folder.parent/'campaign.json'
            if child.exists():
                data=json.loads(child.read_text())
                for previous in data.get('runs',[]):
                    if previous['directory']==folder.name:
                        previous.update(directory=archive.name,status='interrupted')
                atomic(child,data)
        row.pop('pid',None)
        row['status']='pending' if len(row.get('attempts',[]))<=3 else 'error'
        if row['status']=='error': row['termination_category']='repeated_infrastructure_interruptions'
    manifest['status']='stopped_for_inspection'
    atomic(root/'campaign.json',manifest)


def start_service(unit,command,root):
    subprocess.run(['systemd-run','--user','--collect','--unit',unit,
        '--property=Restart=no','--working-directory='+str(PROJECT),
        '--setenv=PYTHONPATH=src',*command],check=True,capture_output=True,text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    with (root/'.watchdog.lock').open('a') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_path=root/'watchdog_status.json'
        previous=json.loads(old_path.read_text()) if old_path.exists() else {}
        status=dict(checked_at=datetime.now(timezone.utc).isoformat(),checked_unix=time.time(),
            state='checking',message='Checking baseline processes',
            restart_history=previous.get('restart_history',[]))
        def report(state,message,**values):
            status.update(state=state,message=message,**values)
            atomic(old_path,status);print(message,flush=True)
        manifest=json.loads((root/'campaign.json').read_text())
        controllers=processes('queue_baseline.py',root)
        active=[r for r in manifest['runs'] if r['status']=='running']
        stalled=[];workers=[]
        for row in active:
            folder=root/row['directory'];pid=row.get('pid')
            if pid and process_matches(pid,'run_local.py',folder.parent):
                workers.append(pid)
                updated=progress_time(folder)
                if updated and time.time()-updated>STALE_SECONDS: stalled.append(pid)
        if controllers and not stalled:
            age=max((time.time()-progress_time(root/r['directory']) for r in active
                if progress_time(root/r['directory'])),default=0)
            report('running','Baseline worker and queue are live',controller_pids=controllers,
                   worker_pids=workers,last_progress_age_s=round(age),active_episodes=[
                    dict(task=r['task_id'],state=r['initial_state_ids'][0]) for r in active])
            return
        if stalled:
            for pid in controllers: stop_tree(pid)
            for pid in stalled:
                if process_matches(pid,'run_local.py'): stop_tree(pid)
            report('recovering','A request exceeded 45 minutes; stopping the affected worker')
            return
        if workers:
            report('recovering','Controller exited; waiting for the live episode to finish',worker_pids=workers)
            return
        if manifest.get('status')=='finished':
            report('finished','Baseline campaign has finished');return
        with (MODEL_ROOT/'.baseline_queue.lock').open('a') as queue_lock:
            try: fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                report('recovering','Queue startup in progress; waiting for its lock');return
            reconcile(root,manifest)
        pending=sum(r['status']=='pending' for r in manifest['runs'])
        if not pending:
            manifest['status']='finished';atomic(root/'campaign.json',manifest)
            report('finished','All baseline episodes have a recorded outcome');return
        recent=[t for t in status['restart_history'] if time.time()-t<3600]
        if len(recent)>=3:
            report('needs_attention','Three restart attempts within an hour; recovery paused to avoid a loop')
            return
        ready=False
        try:
            with httpx.Client(trust_env=False,timeout=5) as client:
                health=client.get('http://127.0.0.1:8081/health')
                if health.status_code==200:
                    props=client.get('http://127.0.0.1:8081/props').json()
                    ready=props['default_generation_settings']['n_ctx']>=131072
        except (httpx.HTTPError,ValueError,KeyError): pass
        if not ready:
            model_pids=processes('llama-server')
            if not model_pids:
                start_service('glimmer-baseline-model-recovery',[
                    '/bin/bash',str(PROJECT/'scripts/serve_model.sh'),'--flash-attn','on',
                    '--cache-type-k','q8_0','--cache-type-v','q8_0',
                    '--chat-template-kwargs','{"reasoning_strength":"xhigh"}'],root)
                status['restart_history'].append(time.time())
            report('recovering','Waiting for the full-context Glimmer model to become ready');return
        start_service('glimmer-baseline-queue-recovery',[
            PYTHON,'-u',str(PROJECT/'scripts/queue_baseline.py'),'--output',str(root),
            '--after-pid','0','--server-pid','0','--resume'],root)
        status['restart_history'].append(time.time())
        report('recovering',f'Restarted the queue for {pending} pending episodes; saved outcomes preserved')


if __name__=='__main__': main()

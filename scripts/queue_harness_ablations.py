"""Sequential, resumable task-0 harness screen; no baseline or paid API changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from llm_agent_memory.harness.experiments import VARIANTS


def atomic(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def metrics(folder):
    path=folder/'episode_00/turns.jsonl';rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    counts={};first_grasp=None;first_completed=None;stagnant=longest=0
    for row in rows:
        try:feedback=json.loads(row['feedback'])
        except (ValueError,TypeError):feedback={}
        status=feedback.get('status','plan_error');counts[status]=counts.get(status,0)+1
        stagnant=stagnant+1 if row['step']==row['start_step'] else 0
        longest=max(longest,stagnant)
        if 'robot.grasp(' in row['response'] and first_grasp is None:first_grasp=row['call']
        if any(e.get('primitive')=='grasp' for e in feedback.get('executed',[])) and first_completed is None:
            first_completed=row['call']
    summary=folder/'summary.json'
    results=json.loads(summary.read_text()).get('results',[]) if summary.exists() else []
    return dict(decisions=len(rows),steps=rows[-1]['step'] if rows else 0,
        statuses=counts,first_grasp_proposed_call=first_grasp,
        first_grasp_primitive_completed_call=first_completed,
        longest_decision_streak_without_physics=longest,
        environment_success=any(r.get('environment_success') for r in results),results=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--campaign-id',required=True)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),
                   default=['control','recovery','clean-images','combined'])
    p.add_argument('--max-calls',type=int,default=80)
    p.add_argument('--max-steps',type=int,default=900)
    p.add_argument('--generation-seed',type=int,default=42)
    p.add_argument('--stale-seconds',type=int,default=1200)
    args=p.parse_args()
    if not args.campaign_id.replace('-','').replace('_','').isalnum():p.error('Use a simple campaign id')
    if min(args.max_calls,args.max_steps,args.stale_seconds)<1:p.error('Limits must be positive')
    if len(set(args.variants))!=len(args.variants):p.error('Duplicate variant')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.ablation_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=root/(args.campaign_id+'.json')
    project=Path(__file__).resolve().parents[1]
    protocol=dict(task_id=0,initial_state=0,effort='high',profile='wrist-v3',
        max_calls=args.max_calls,max_steps=args.max_steps,max_output_tokens=16384,
        generation_seed=args.generation_seed,variants=args.variants,
        switches={k:VARIANTS[k] for k in args.variants},
        generalizable=True,task_specific_prompts=False,paid_api=False,
        interpretation='One state per variant is a diagnostic screen, not a success-rate estimate.')
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(protocol=protocol,trials=[])
    if state['protocol']!=protocol:raise SystemExit('Existing campaign has a different protocol')
    child=None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for variant in args.variants:
            previous=next((r for r in state['trials'] if r['variant']==variant),None)
            if previous:
                if previous['status']=='completed':continue
                raise RuntimeError('Interrupted trial requires inspection; no automatic rerun: '+variant)
            label=args.campaign_id+'-'+variant
            trial=dict(variant=variant,label=label,status='running',started_at=time.time())
            state['trials'].append(trial);atomic(statepath,state)
            command=[sys.executable,'-u',str(project/'scripts/run_harness.py'),
                '--output',str(root),'--append','--trial-label',label,
                '--variant',variant,'--inits','0','--effort','high','--profile','wrist-v3',
                '--max-calls',str(args.max_calls),'--max-steps',str(args.max_steps),
                '--max-output-tokens','16384','--generation-seed',str(args.generation_seed)]
            print('Starting '+label,flush=True)
            child=subprocess.Popen(command,cwd=project,start_new_session=True)
            trial['pid']=child.pid;stale=False
            while child.poll() is None:
                now=time.time();trial['checked_at']=now
                manifest=root/'campaign.json'
                if manifest.exists():
                    run=next((r for r in json.loads(manifest.read_text())['runs'] if r.get('trial_label')==label),None)
                    if run:
                        folder=root/run['directory'];trial['directory']=run['directory']
                        trial['metrics']=metrics(folder)
                        files=[folder/'api/api_usage.jsonl',folder/'episode_00/turns.jsonl',folder/'protocol.json']
                        last=max((x.stat().st_mtime for x in files if x.exists()),default=trial['started_at'])
                        trial['last_progress_at']=last
                    else:last=trial['started_at']
                else:last=trial['started_at']
                atomic(statepath,state)
                if now-last>args.stale_seconds:
                    stale=True;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(10)
            trial['exit_code']=child.wait();trial['finished_at']=time.time()
            if 'directory' in trial:trial['metrics']=metrics(root/trial['directory'])
            trial['status']='completed' if trial['exit_code']==0 and not stale else 'interrupted'
            atomic(statepath,state)
            if trial['status']!='completed':raise RuntimeError('Trial interrupted; queue paused for inspection')
        state['status']='completed'
    except BaseException:
        state['status']='interrupted'
        if state['trials'] and state['trials'][-1]['status']=='running':state['trials'][-1]['status']='interrupted'
        raise
    finally:
        state['checked_at']=time.time();atomic(statepath,state)


if __name__=='__main__':main()

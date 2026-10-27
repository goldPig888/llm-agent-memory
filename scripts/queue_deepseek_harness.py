"""Run three max-effort DeepSeek harness episodes after visual trials finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--env-file',type=Path,required=True)
    args=p.parse_args();root=args.output.resolve()
    deadline=time.monotonic()+12*3600
    status=root/'deepseek_queue_status.json'
    def save(phase):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(
            phase=phase,checked_at=time.time(),pid=os.getpid(),model='deepseek/deepseek-v4.1-flash',
            effort='max',task_id=0,states=[0,1,2],variant='clean-images',batch_budget_usd=5),indent=2))
        temp.replace(status)
    while time.monotonic()<deadline:
        path=root/'visual-ablation-v1.json'
        data=json.loads(path.read_text()) if path.exists() else {}
        active=subprocess.run(['systemctl','--user','is-active','--quiet','glimmer-visual-ablations.service']).returncode==0
        if not active:
            complete=data.get('status')=='completed' and len(data.get('trials',[]))==3 and all(
                t['status']=='completed' for t in data.get('trials',[]))
            if not complete:
                save('blocked_visual_queue_incomplete')
                raise SystemExit('Visual queue stopped incomplete; DeepSeek was not started')
            break
        save('waiting_for_visual_trials');time.sleep(15)
    else:
        save('wait_timeout');raise SystemExit('Visual trial wait timed out; nothing started')
    project=Path(__file__).resolve().parents[1];os.chdir(project)
    save('starting_deepseek')
    command=[sys.executable,'-u',str(project/'scripts/run_harness.py'),
        '--output',str(root),'--append','--trial-label','deepseek-max-clean-states012',
        '--provider','haimaker','--env-file',str(args.env_file.resolve()),'--api-budget','5',
        '--profile','wrist-v3','--variant','clean-images','--effort','max',
        '--inits','0','1','2','--max-calls','80','--max-steps','900','--max-output-tokens','16384']
    print('Visual trials complete. Starting three DeepSeek max episodes; batch cap $5.',flush=True)
    os.execv(sys.executable,command)


if __name__=='__main__':main()

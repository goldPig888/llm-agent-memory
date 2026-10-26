"""Run one labeled robot trial only after the local pointing service completes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--label',default='wrist-v3-high-state0')
    args=p.parse_args();deadline=time.monotonic()+6*3600
    protocol=json.loads((args.results/'protocol.json').read_text())
    expected=protocol['samples']*len(protocol['models'])
    while time.monotonic()<deadline:
        active=subprocess.run(['systemctl','--user','is-active','--quiet','glimmer-pointing-eval.service']).returncode==0
        path=args.results/'summary.json'
        results=json.loads(path.read_text()).get('results',[]) if path.exists() else []
        complete=len({(r['model'],r['id']) for r in results})==expected
        if not active:
            if not complete:raise SystemExit('Pointing service ended before all samples; robot trial was not started')
            break
        time.sleep(15)
    else:raise SystemExit('Pointing completion wait timed out; robot trial was not started')
    project=Path(__file__).resolve().parents[1];os.chdir(project)
    command=[sys.executable,'-u',str(project/'scripts/run_harness.py'),
        '--output',str(args.output.resolve()),'--append','--trial-label',args.label,
        '--profile','wrist-v3','--inits','0','--effort','high','--max-calls','80',
        '--max-steps','900','--max-output-tokens','16384']
    print('Pointing diagnostic complete; starting one wrist-v3 Task 0/state 0 trial.',flush=True)
    os.execv(sys.executable,command)


if __name__=='__main__':main()

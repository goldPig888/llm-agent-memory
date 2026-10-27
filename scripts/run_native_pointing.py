"""Paired xhigh native-box vs normalized-point diagnostic on saved RGB images."""
import argparse
import ast
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import httpx
import numpy as np

from run_pointing import input_message,parse_point,score_point,save_summary


NATIVE_PROMPT=("Detect the alphabet soup can. Return only the detection in the model's "
               "native object-detection format, with no explanation.")


def native_message(sample,root):
    # Dataset geometry, masks, and label metadata are never accepted as input.
    path=(root/sample['image']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Image outside dataset')
    encoded=base64.b64encode(path.read_bytes()).decode()
    return dict(role='user',content=[dict(type='text',text=NATIVE_PROMPT),
        dict(type='image_url',image_url=dict(url='data:image/png;base64,'+encoded))])


def parse_native_box(text):
    keys=('x_min','y_min','x_max','y_max')
    for candidate in reversed(re.findall(r'\{[^{}]+\}',text)):
        try:value=json.loads(candidate)
        except ValueError:
            try:value=ast.literal_eval(candidate)
            except (ValueError,SyntaxError):continue
        if not isinstance(value,dict) or not all(k in value for k in keys):continue
        values=[value[k] for k in keys]
        if not all(type(x) in (int,float) and np.isfinite(x) for x in values):
            raise ValueError('Box coordinates must be finite numbers')
        if not all(0<=x<=1000 for x in values):raise ValueError('Box outside assumed 0..1000 grid')
        x0,y0,x1,y1=map(float,values)
        if not x0<x1 or not y0<y1:raise ValueError('Box corners reversed or zero area')
        return dict(zip(keys,(x0,y0,x1,y1))),[(x0+x1)/2000,(y0+y1)/2000]
    raise ValueError('No native box with x_min/y_min/x_max/y_max found')


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--states',type=int,nargs='+',default=[0,1,2,3,4],choices=range(50))
    p.add_argument('--wait-campaign',type=Path)
    p.add_argument('--wait-service',default='glimmer-visual-xhigh-ablations.service')
    args=p.parse_args()
    if len(set(args.states))!=len(args.states):p.error('Duplicate states')
    root=args.dataset.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    samples=[s for s in map(json.loads,(root/'samples.jsonl').read_text().splitlines())
        if s['state'] in args.states and s['camera']=='agentview']
    samples.sort(key=lambda s:args.states.index(s['state']))
    if len(samples)!=len(args.states):raise ValueError('Missing requested state')
    protocol=dict(model='muse-glimmer',effort='xhigh',max_output_tokens=16384,
        states=args.states,cameras=['agentview'],samples=len(samples)*2,
        formats=['normalized-point','native-box'],coordinate_scale_assumption=1000,
        native_prompt=NATIVE_PROMPT,seed='42 + state; same for paired formats',
        order='point first on even-index samples; native first on odd-index samples',
        images='Unmodified saved RGB; identical bytes for each pair',physics_steps=0,
        labels='Scoring only; never included in messages',
        interpretation='Small paired pilot, not a 50-state result. Historical 44% used high, not xhigh.',
        sources=['https://gist.github.com/ariG23498/4f3587eb7753c0ff77c269e2c1efe2c0',
                 'https://huggingface.co/blog/muse-glimmer',
                 'https://huggingface.co/merve/muse-glimmer-ft-clicking-v2'])
    atomic(output/'protocol.json',protocol)
    progress=dict(phase='initializing',checked_at=time.time(),pid=os.getpid(),completed=0,total=len(samples)*2)
    def status(phase):
        progress.update(phase=phase,checked_at=time.time());atomic(output/'progress.json',progress)
    deadline=time.monotonic()+12*3600
    try:
        if args.wait_campaign:
            while time.monotonic()<deadline:
                active=subprocess.run(['systemctl','--user','is-active','--quiet',args.wait_service]).returncode==0
                state=json.loads(args.wait_campaign.read_text())
                if not active:
                    if state.get('status')!='completed':raise RuntimeError('Preceding campaign incomplete; no model calls sent')
                    break
                status('waiting_for_xhigh_robot_trials');time.sleep(15)
            else:raise RuntimeError('Waiting for robot trials timed out')
        model_root=Path(os.environ.get('MUSE_ROOT','/metadisk/may/muse-glimmer'))
        with (model_root/'.baseline_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status('running');rows=[]
            with httpx.Client(trust_env=False,timeout=1200) as client:
                props=client.get('http://127.0.0.1:8081/props');props.raise_for_status()
                protocol['server_build']=props.json().get('build_info')
                protocol['server_generation_defaults']=props.json().get('default_generation_settings')
                atomic(output/'protocol.json',protocol)
                for index,sample in enumerate(samples):
                    modes=['normalized-point','native-box'] if index%2==0 else ['native-box','normalized-point']
                    for mode in modes:
                        message=native_message(sample,root) if mode=='native-box' else input_message(sample,root)
                        body=dict(model='muse-glimmer',messages=[message],max_tokens=16384,
                            seed=42+sample['state'],chat_template_kwargs={'reasoning_strength':'xhigh'})
                        ident=sample['id']+'_'+mode
                        atomic(output/(ident+'_request.json'),body)
                        started=time.monotonic();status('running')
                        response=client.post('http://127.0.0.1:8081/v1/chat/completions',json=body)
                        (output/(ident+'_response.json')).write_bytes(response.content);response.raise_for_status()
                        payload=response.json();choice=payload['choices'][0];text=choice['message'].get('content') or ''
                        point=None;box=None;error=None
                        try:
                            if choice.get('finish_reason')=='length':raise ValueError('Output truncated')
                            if mode=='native-box':box,point=parse_native_box(text)
                            else:point=parse_point(text)
                        except (ValueError,KeyError,TypeError) as exc:error=str(exc)
                        # Labels loaded only for scoring after the request completes.
                        label=json.loads((root/'scoring/labels.json').read_text())[sample['id']]
                        row=dict(model=mode,id=sample['id'],state=sample['state'],camera=sample['camera'],
                            prediction=point,bbox=box,parse_error=error,raw_output=text,
                            scale_warning='Box values all <= 1; 1000 scale ambiguous' if box and max(box.values())<=1 else None,
                            score=score_point(point,label,root),latency_s=time.monotonic()-started,
                            image_sha256=hashlib.sha256((root/sample['image']).read_bytes()).hexdigest(),
                            usage=payload.get('usage',{}))
                        rows.append(row);save_summary(output,rows,0)
                        progress['completed']=len(rows);status('running')
                        print(json.dumps({k:row[k] for k in ['model','id','prediction','bbox','score','parse_error']}),flush=True)
            status('completed')
    except BaseException:
        status('interrupted');raise


if __name__=='__main__':main()

"""Pointing-only comparison. Images/instructions are the only task inputs sent."""
import argparse
import base64
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import time

import httpx
import numpy as np

from inspect_robots._dotenv import read_dotenv


def input_message(sample,root):
    # Deliberately do not accept scoring labels or camera geometry here.
    image_path=(root/sample['image']).resolve()
    if not image_path.is_relative_to(root.resolve()):raise ValueError('Image outside dataset')
    data=base64.b64encode(image_path.read_bytes()).decode()
    text=(sample['instruction']+' Return only JSON {"u": number, "v": number}. '
        'Coordinates are normalized 0..1, with (0,0) at image top-left. '
        'If the target is not visible return {"visible": false}. Do not guess coordinates for an invisible target.')
    return dict(role='user',content=[dict(type='text',text=text),
        dict(type='image_url',image_url=dict(url='data:image/png;base64,'+data))])


def parse_point(text):
    match=re.search(r'\{[^{}]*\}',text)
    if not match:raise ValueError('No JSON object')
    value=json.loads(match.group())
    if value.get('visible') is False:return None
    u,v=value['u'],value['v']
    if not all(type(x) in (float,int) and np.isfinite(x) and 0<=x<=1 for x in (u,v)):
        raise ValueError('Coordinates must be finite numbers in 0..1')
    return [float(u),float(v)]


def score_point(point,label,root):
    if not label['visible_pixels']:
        return dict(target_visible=False,hit_visible_object=None,pixel_error=None)
    if point is None:
        return dict(target_visible=True,hit_visible_object=False,pixel_error=None)
    mask=np.load(root/label['mask']);u,v=point;pixel=np.array([u*(mask.shape[1]-1),v*(mask.shape[0]-1)])
    hit=bool(mask[int(round(pixel[1])),int(round(pixel[0]))])
    center=label.get('projected_center_px')
    return dict(target_visible=True,hit_visible_object=hit,
        pixel_error=float(np.linalg.norm(pixel-center)) if center is not None else None)


def save_summary(output,rows,cost):
    groups={}
    for row in rows:
        key=row['model']+'/'+row['camera'];g=groups.setdefault(key,Counter())
        g['samples']+=1;g['visible']+=bool(row['score']['target_visible'])
        g['hits']+=bool(row['score']['hit_visible_object']);g['parse_errors']+=bool(row.get('parse_error'))
    summary=dict(groups={k:dict(v,hit_rate=v['hits']/v['visible'] if v['visible'] else None) for k,v in groups.items()},
        api_cost_upper_bound_usd=cost,physics_steps=0,results=rows)
    temp=output/'summary.tmp';temp.write_text(json.dumps(summary,indent=2));temp.replace(output/'summary.json')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--models',nargs='+',choices=['glimmer','haimaker'],default=['glimmer'])
    p.add_argument('--effort',choices=['low','high','xhigh'],default='high')
    p.add_argument('--max-output-tokens',type=int,default=4096);p.add_argument('--limit',type=int)
    p.add_argument('--env-file',type=Path);p.add_argument('--total-budget',type=float,default=5)
    p.add_argument('--execute-api',action='store_true')
    args=p.parse_args()
    if not 0<args.total_budget<=5: p.error('API budget must be >0 and <=5 USD')
    if args.max_output_tokens<1 or args.max_output_tokens>16384:p.error('Output cap must be 1..16384')
    settings={**(read_dotenv(args.env_file) if args.env_file else {}),**os.environ}
    if 'haimaker' in args.models and (not args.execute_api or not settings.get('HAIMAKER_API_KEY')):
        p.error('Haimaker requires --execute-api and HAIMAKER_API_KEY (optionally via --env-file)')
    if 'haimaker' in args.models and args.effort=='xhigh':p.error('Use common high effort for this comparison')
    root=args.dataset.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    samples=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    if args.limit:samples=samples[:args.limit]
    labels=json.loads((root/'scoring/labels.json').read_text())
    model_root=Path(os.environ.get('MUSE_ROOT',
        '/metadisk/may/muse-glimmer' if Path('/metadisk/may/muse-glimmer').exists()
        else str(Path.home()/'.local/share/muse-glimmer')))
    rows=[];used=0.
    with (model_root/'.baseline_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (output/'protocol.json').write_text(json.dumps(dict(models=args.models,effort=args.effort,
            max_output_tokens=args.max_output_tokens,samples=len(samples),physics_steps=0,
            privileged_labels='scoring only; not transmitted',api_budget_usd=args.total_budget),indent=2))
        for model in args.models:
            endpoint='http://127.0.0.1:8081/v1/chat/completions' if model=='glimmer' else 'https://api.haimaker.ai/v1/chat/completions'
            headers={} if model=='glimmer' else {'Authorization':'Bearer '+settings['HAIMAKER_API_KEY']}
            with httpx.Client(trust_env=False,timeout=600,headers=headers) as client:
                for sample in samples:
                    reserve=(1048576*.3+args.max_output_tokens*1.2)/1e6 if model=='haimaker' else 0
                    if used+reserve>args.total_budget:raise RuntimeError('API budget reservation limit; requests stopped')
                    body=dict(model='muse-glimmer' if model=='glimmer' else 'deepseek/deepseek-v4.1-flash',
                        messages=[input_message(sample,root)],max_tokens=args.max_output_tokens)
                    if model=='glimmer':body['chat_template_kwargs']={'reasoning_strength':args.effort}
                    else:body['reasoning_effort']=args.effort
                    ident=model+'_'+sample['id'];(output/(ident+'_request.json')).write_text(json.dumps(body))
                    started=time.monotonic();response=client.post(endpoint,json=body)
                    (output/(ident+'_response.json')).write_bytes(response.content);response.raise_for_status()
                    payload=response.json();usage=payload.get('usage') or {}
                    if model=='haimaker':
                        inp,out=usage.get('prompt_tokens'),usage.get('completion_tokens')
                        if not all(type(v) is int and v>=0 for v in (inp,out)):
                            raise RuntimeError('API usage unknown; no further requests sent')
                        used+=(inp*.3+out*1.2)/1e6
                    choice=payload['choices'][0];text=choice['message'].get('content') or '';point=None;error=None
                    try:
                        if choice.get('finish_reason')=='length':raise ValueError('Output truncated')
                        point=parse_point(text)
                    except (ValueError,KeyError,TypeError) as exc:error=str(exc)
                    rows.append(dict(model=model,id=sample['id'],state=sample['state'],camera=sample['camera'],
                        prediction=point,parse_error=error,score=score_point(point,labels[sample['id']],root),
                        latency_s=time.monotonic()-started,usage=usage))
                    save_summary(output,rows,used)
                    print(json.dumps({k:rows[-1][k] for k in ['model','id','prediction','score','parse_error']}),flush=True)
    print(json.dumps(save_summary(output,rows,used)['groups']),flush=True)


if __name__=='__main__':main()

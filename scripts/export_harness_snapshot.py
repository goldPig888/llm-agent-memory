"""Export portable harness summaries and per-turn programs without raw API traffic."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness',type=Path,required=True)
    p.add_argument('--pointing',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True)
    args=p.parse_args();args.docs.mkdir(parents=True,exist_ok=True)
    captured=datetime.now(timezone.utc).isoformat()
    campaigns=[]
    for path in sorted(args.harness.glob('*-ablation-*.json')):
        source=json.loads(path.read_text());trials={t['variant']:t for t in source['trials']}
        portable=[]
        for variant in source['protocol']['variants']:
            t=trials.get(variant,{})
            metrics={k:v for k,v in t.get('metrics',{}).items() if k!='results'}
            results=[{k:v for k,v in r.items() if k!='video'} for r in t.get('metrics',{}).get('results',[])]
            portable.append(dict(variant=variant,status=t.get('status','queued'),
                directory=t.get('directory'),metrics=metrics,results=results))
        campaigns.append(dict(name=path.stem,status=source.get('status','running'),
            protocol=source['protocol'],trials=portable))
    output=dict(captured_utc=captured,snapshot_only=True,
        note='Running/queued rows are partial, not completed failures. These are single-state screens.',campaigns=campaigns)
    (args.docs/'harness_ablation_results.json').write_text(json.dumps(output,indent=2)+'\n')
    snapshot=json.loads((args.docs/'results_snapshot.json').read_text())
    snapshot['captured_utc']=captured;snapshot['harness_trials']=[]
    turns=[]
    for run in json.loads((args.harness/'campaign.json').read_text())['runs']:
        folder=args.harness/run['directory']
        protocol=json.loads((folder/'protocol.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        label=run.get('trial_label') or run['directory']
        profile=protocol.get('execution_profile','original')
        results=[{k:v for k,v in r.items() if k!='video'} for r in summary.get('results',[])]
        snapshot['harness_trials'].append(dict(trial=label,queue_status=run['status'],
            profile=profile,effort=protocol.get('reasoning_effort'),
            variant=protocol.get('experiment_variant'),context=protocol.get('server_context_tokens'),results=results))
        for episode,state in enumerate(protocol['initial_state_ids']):
            path=folder/f'episode_{episode:02d}/turns.jsonl'
            if not path.exists():continue
            for line in path.read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                feedback=row['feedback']
                try:status=json.loads(feedback).get('status','unknown')
                except (ValueError,TypeError):status=feedback
                # Preserve the exact program, but not image-containing requests.
                from llm_agent_memory.harness.upstream import extract_curl
                try:
                    endpoint,code=extract_curl(row['response']);program=code or endpoint
                except ValueError:program=row['response']
                turns.append(dict(trial=label,profile=profile,state=state,call=row['call'],
                    start_step=row['start_step'],end_step=row['step'],steps_used=row['step']-row['start_step'],
                    feedback_status=status,program=program))
    snapshot['harness_ablation_results']='harness_ablation_results.json'
    snapshot['pointing_results']='pointing_results_snapshot.json'
    (args.docs/'results_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    with (args.docs/'harness_turns_snapshot.csv').open('w',newline='') as handle:
        fields=['trial','profile','state','call','start_step','end_step','steps_used','feedback_status','program']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(turns)
    pointing=json.loads((args.pointing/'summary.json').read_text())
    groups=pointing['groups']
    for key,value in groups.items():
        camera=key.split('/')[-1]
        value['abstentions']=sum(r['prediction'] is None for r in pointing['results'] if r['camera']==camera)
    (args.docs/'pointing_results_snapshot.json').write_text(json.dumps(dict(
        captured_utc=captured,snapshot_only=True,groups=groups,samples=len(pointing['results']),
        physics_steps=pointing['physics_steps'],api_cost_upper_bound_usd=pointing['api_cost_upper_bound_usd'],
        note='Visible-object mask hits, not manipulation success. Wrist images are from the initial pose.'),indent=2)+'\n')
    lines=['# Latest harness results','',f'Snapshot captured at {captured}.',
        'These are single-state diagnostic trials, not success-rate estimates.',
        'All variants preserve the general prompt; no task-specific solution prompt was added.','',
        '| Campaign / variant | Status | Decisions | Physics steps | First grasp proposed | Task outcome |',
        '|---|---|---:|---:|---:|---|']
    for c in campaigns:
        for t in c['trials']:
            m=t['metrics'];outcome='success' if any(r.get('environment_success') for r in t['results']) else ('failure' if t['results'] else 'pending')
            lines.append(f"| {c['name']} / {t['variant']} | {t['status']} | {m.get('decisions','—')} | {m.get('steps','—')} | {m.get('first_grasp_proposed_call') or '—'} | {outcome} |")
    lines+=['', 'The completed four-way screen had no full grasp primitive completion or task success.',
        'Clean images reached a grasp proposal at decision 3. Combined preprocessing/recovery',
        'produced seven dropped outcomes during grasp/lift attempts; this does not establish',
        'correct-object retention. Recovery blocked four identical rejected retries.',
        'The visual follow-up compares raw clean RGB, gamma 0.8, and contrast normalization.',
        'Rows still running/queued at capture have no final outcome.','',
        'The completed original-policy xhigh baseline remains 0/100 successes (99 failures, 1 error).',
        'The completed pointing diagnostic hit the target mask in 22/50 agentview cases and',
        '9/50 initial-wrist cases, with 8 and 41 abstentions respectively.','',
        '## Overlay audit and next experiments','',
        'The grid/reticle design follows the upstream report, but the LIBERO renderer is our',
        'implementation. Our dense lines overlap labels; coordinate text is small; the prompt',
        'describes yellow x/cyan y while the renderer currently draws both in yellow.',
        'Proposed next comparisons are reticle-only and a sparse faint grid with a consistent',
        'legend. These overlay redesigns have **not** been implemented or run yet.','',
        '## Review on another PC','',
        'Pull this repository and read this file, HARNESS_ABLATIONS.md, harness_ablation_results.json,',
        'pointing_results_snapshot.json, and harness_turns_snapshot.csv. No local model or simulator',
        'is needed to read the summaries and programs. These committed files are a dated snapshot,',
        'not a live copy of the experiment queue. Videos, observations, weights, and raw API requests',
        'remain on the experiment machine and are not included in Git.','']
    (args.docs/'LATEST_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(captured_utc=captured,campaigns=len(campaigns),turns=len(turns))))


if __name__=='__main__':main()

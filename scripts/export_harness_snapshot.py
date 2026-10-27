"""Export portable harness summaries and per-turn programs without raw API traffic."""
import argparse
import csv
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness',type=Path,required=True)
    p.add_argument('--pointing',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True)
    p.add_argument('--native',type=Path,help='Optional native-format pilot output directory')
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
    turns=[];deepseek=[]
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
        episode_metrics=[]
        for episode,state in enumerate(protocol['initial_state_ids']):
            path=folder/f'episode_{episode:02d}/turns.jsonl'
            if not path.exists():continue
            executed=Counter();grasp_calls=[];place_calls=[];first_proposal=None
            for line in path.read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                feedback=row['feedback']
                try:
                    parsed=json.loads(feedback);status=parsed.get('status','unknown')
                except (ValueError,TypeError):status=feedback;parsed={}
                if first_proposal is None and 'robot.grasp(' in row['response']:first_proposal=row['call']
                for e in parsed.get('executed',[]):
                    executed[e['primitive']]+=1
                    if e['primitive']=='grasp':grasp_calls.append(row['call'])
                    if e['primitive'] in {'place','place_relative','place_until_contact'}:place_calls.append(row['call'])
                # Preserve the exact program, but not image-containing requests.
                from llm_agent_memory.harness.upstream import extract_curl
                try:
                    endpoint,code=extract_curl(row['response']);program=code or endpoint
                except ValueError:program=row['response']
                turns.append(dict(trial=label,profile=profile,state=state,call=row['call'],
                    start_step=row['start_step'],end_step=row['step'],steps_used=row['step']-row['start_step'],
                    feedback_status=status,program=program))
            episode_metrics.append(dict(state=state,executed_primitives=dict(executed),
                first_grasp_proposed_call=first_proposal,completed_grasp_calls=grasp_calls,
                completed_place_calls=place_calls))
        if protocol.get('provider')=='haimaker':
            budget_path=folder/'api_budget.json'
            deepseek.append(dict(trial=label,status=run['status'],model=protocol['model'],
                requested_effort=protocol['reasoning_effort'],variant=protocol['experiment_variant'],
                max_calls=protocol['max_api_attempts_per_episode'],max_steps=protocol['max_steps'],
                max_output_tokens=protocol['max_output_tokens'],results=results,milestones=episode_metrics,
                budget=json.loads(budget_path.read_text()) if budget_path.exists() else None))
    snapshot['harness_ablation_results']='harness_ablation_results.json'
    snapshot['pointing_results']='pointing_results_snapshot.json'
    snapshot['deepseek_results']='deepseek_harness_results.json'
    (args.docs/'deepseek_harness_results.json').write_text(json.dumps(dict(
        captured_utc=captured,snapshot_only=True,trials=deepseek,
        note='Completed primitives do not prove correct-object retention or placement. Costs are list-price bounds.'),indent=2)+'\n')
    native=None
    if args.native and (args.native/'protocol.json').exists():
        state=json.loads((args.native/'progress.json').read_text())
        summary_path=args.native/'summary.json'
        native=dict(captured_utc=captured,snapshot_only=True,
            protocol=json.loads((args.native/'protocol.json').read_text()),
            progress={k:state.get(k) for k in ['phase','checked_at','completed','total']},
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else None)
        (args.docs/'native_pointing_results.json').write_text(json.dumps(native,indent=2)+'\n')
        snapshot['native_pointing_results']='native_pointing_results.json'
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
        'Glimmer ablations are single-state screens; DeepSeek covers three states. Neither establishes a success rate.',
        'All variants preserve the general prompt; no task-specific solution prompt was added.','',
        '| Campaign / variant | Effort | Status | Decisions | Physics steps | First grasp proposed | Task outcome |',
        '|---|---|---|---:|---:|---:|---|']
    for c in campaigns:
        for t in c['trials']:
            m=t['metrics'];outcome='success' if any(r.get('environment_success') for r in t['results']) else ('failure' if t['results'] else 'pending')
            lines.append(f"| {c['name']} / {t['variant']} | {c['protocol']['effort']} | {t['status']} | {m.get('decisions','—')} | {m.get('steps','—')} | {m.get('first_grasp_proposed_call') or '—'} | {outcome} |")
    lines+=['', 'The completed four-way screen had no full grasp primitive completion or task success.',
        'Clean images reached a grasp proposal at decision 3. Combined preprocessing/recovery',
        'produced seven dropped outcomes during grasp/lift attempts; this does not establish',
        'correct-object retention. Recovery blocked four identical rejected retries.',
        'The visual follow-up compares raw clean RGB, gamma 0.8, and contrast normalization.',
        'Rows still running/queued at capture have no final outcome.','',
        'The completed original-policy xhigh baseline remains 0/100 successes (99 failures, 1 error).',
        'The completed pointing diagnostic hit the target mask in 22/50 agentview cases and',
        '9/50 initial-wrist cases, with 8 and 41 abstentions respectively.','',
        '## DeepSeek max-effort harness comparison','',
        'Three Task 0 states (0,1,2), clean images, 80 calls/900 steps per episode,',
        '16,384 output tokens, Haimaker with max requested. Internal provider effort',
        'is not independently observable. See deepseek_harness_results.json.','',
        '| Trial / state | Status | Calls | Steps | Completed grasps | Completed placements |',
        '|---|---|---:|---:|---:|---:|']
    for t in deepseek:
        milestones={m['state']:m for m in t['milestones']}
        for r in t['results']:
            m=milestones.get(r['init_state'],{})
            lines.append(f"| {t['trial']} / {r['init_state']} | {r['status']} | {r['api_attempts']} | {r['steps']} | {len(m.get('completed_grasp_calls',[]))} | {len(m.get('completed_place_calls',[]))} |")
        if t['budget']:
            lines+=['',f"Estimated peak-list-price bound: ${t['budget']['charged_upper_bound_usd']:.4f}; not a promotional invoice."]
    lines+=['','Completed grasp/place primitives are intermediate milestones; the LIBERO',
        'predicate determines full-task success. They do not verify target identity.','',
        '## Native-format pointing pilot','']
    if native:
        phase=native['progress']['phase'];completed=native['progress']['completed'];total=native['progress']['total']
        lines+=[f'Snapshot phase: **{phase}**, {completed}/{total} requests recorded.',
            'States 0–4, agent camera, xhigh. Paired normalized-point vs native-box prompts.',
            'Primary box center uses the predeclared 0–1000 assumption; raw outputs are retained.',
            'See NATIVE_POINTING.md and native_pointing_results.json. No result is claimed while waiting.','']
    else:lines+=['No native-format pilot snapshot supplied.','']
    lines+=[
        '## Overlay audit and next experiments','',
        'The grid/reticle design follows the upstream report, but the LIBERO renderer is our',
        'implementation. Our dense lines overlap labels; coordinate text is small; the prompt',
        'describes yellow x/cyan y while the renderer currently draws both in yellow.',
        'Proposed next comparisons are reticle-only and a sparse faint grid with a consistent',
        'legend. These overlay redesigns have **not** been implemented or run yet.','',
        '## Review on another PC','',
        'Pull this repository and read this file, HARNESS_ABLATIONS.md, harness_ablation_results.json,',
        'pointing_results_snapshot.json, deepseek_harness_results.json, native_pointing_results.json,',
        'and harness_turns_snapshot.csv. No local model or simulator',
        'is needed to read the summaries and programs. These committed files are a dated snapshot,',
        'not a live copy of the experiment queue. Videos, observations, weights, and raw API requests',
        'remain on the experiment machine and are not included in Git.','']
    (args.docs/'LATEST_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(captured_utc=captured,campaigns=len(campaigns),turns=len(turns))))


if __name__=='__main__':main()

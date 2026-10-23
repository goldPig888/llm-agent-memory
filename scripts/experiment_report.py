"""Local Muse Glimmer LIBERO dashboard, adapted from IRVLUTD/LLM_LIBERO."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from inspect_robots_libero.tasks import LIBERO_10_TASKS


def collect(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / "campaign.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    annotation_path = root / "manual_annotations.json"
    annotations = json.loads(annotation_path.read_text()).get("episodes", []) if annotation_path.exists() else []
    reviewed = {(item["directory"], int(item["init_state"])): item for item in annotations}
    episodes = []
    for run in manifest.get("runs", []):
        if run.get("status") == "superseded":
            continue  # Keep the interrupted evidence in the manifest without duplicate dashboard rows.
        folder = root / run["directory"]
        summary_path = folder / "summary.json"
        protocol_path = folder / "protocol.json"
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"results": []}
        by_episode = {int(r["episode"]): r for r in summary.get("results", [])}
        events_path = folder / "api" / "api_usage.jsonl"
        events = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # A process may still be writing its last line.
        resume_from = protocol.get('resume_from')
        if resume_from:
            source_events = Path(resume_from) / 'api/api_usage.jsonl'
            if source_events.exists():
                previous = [json.loads(line) for line in source_events.read_text().splitlines()]
                events = [dict(event, budget_charge_usd=event['estimated_usd'])
                          if event.get('estimated_usd') is not None else event
                          for event in previous] + events
        call_events = {}
        for event in events:
            if event.get("event") in {"request_started", "response", "usage_unknown"}:
                key = (event.get("episode"), event.get("call"))
                call_events.setdefault(key, {}).update({event["event"]: event})
        state_ids = protocol.get("initial_state_ids", run.get("initial_state_ids", []))
        for episode, state_id in enumerate(state_ids):
            if state_id in run.get("superseded_initial_state_ids", []):
                continue
            result = by_episode.get(episode, {})
            # Older runners grouped call exhaustion with dollar-budget stops.
            if (result.get("status") == "incomplete_budget"
                    and not result.get("stop_detail")
                    and result.get("api_attempts", 0) >= protocol.get(
                        "max_api_attempts_per_episode", run.get("max_calls", 80))):
                result = dict(result, status="failure", stop_detail="call_limit")
            review = reviewed.get((run["directory"], state_id), {})
            calls = []
            for (ep, call_no), stages in sorted(call_events.items()):
                if ep != episode:
                    continue
                started = stages.get("request_started", {})
                response = stages.get("response", {})
                usage = response.get("usage") or {}
                response_path = folder / "api" / f"ep{episode:02d}_call{call_no:02d}_response.json"
                if not response_path.exists() and resume_from:
                    response_path = Path(resume_from) / 'api' / response_path.name
                payload = {}
                if response_path.exists():
                    try:
                        payload = json.loads(response_path.read_text())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                if protocol.get("wire") == "messages":
                    tools = [b.get("name") for b in payload.get("content", []) if b.get("type") == "tool_use"]
                    truncated = payload.get("stop_reason") == "max_tokens"
                else:
                    tools = [c.get("function", {}).get("name") for ch in payload.get("choices", []) for c in ch.get("message", {}).get("tool_calls", [])] or [b.get("name") for b in payload.get("output", []) if b.get("type") == "function_call"]
                    truncated = any(ch.get("finish_reason") == "length" for ch in payload.get("choices", [])) or (payload.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
                calls.append({"call": call_no, "input_tokens": usage.get("input_tokens"),
                    "response_text": '\n'.join((ch.get('message', {}).get('content') or '') + '\n' +
                        '\n'.join(c.get('function', {}).get('name', '') + ': ' + c.get('function', {}).get('arguments', '')
                                  for c in ch.get('message', {}).get('tool_calls', []))
                        for ch in payload.get('choices', [])),
                    "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", usage.get("cache_read_input_tokens", 0)),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", (usage.get("output_tokens_details") or {}).get("thinking_tokens")),
                    "estimated_usd": response.get("estimated_usd"), "latency_s": response.get("latency_s"),
                    "http_status": response.get("http_status"), "tools": tools,
                    "truncated": truncated, "reserved_usd": started.get("reservation_usd"),
                    "usage_unknown": "usage_unknown" in stages})
            inspect_dir = folder / f"episode_{episode:02d}" / "inspect"
            trajectory = []
            for action_file in (inspect_dir / "actions").rglob("*.jsonl"):
                if f"init{state_id:02d}" not in action_file.name:
                    continue
                for line in action_file.read_text().splitlines():
                    try:
                        action_row = json.loads(line)
                        action = action_row.get("action")
                        if isinstance(action, list) and len(action) == 10:
                            trajectory.append([int(action_row["t"]), *action[:3], action[9]])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            conversation = []
            for transcript_file in (inspect_dir / "transcripts").rglob("*.jsonl"):
                if f"init{state_id:02d}" not in transcript_file.name:
                    continue
                for line in transcript_file.read_text().splitlines():
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    role = message.get("role")
                    if role == "system":
                        continue  # The exact prompt is recorded once in the protocol.
                    content = message.get("content")
                    if isinstance(content, list):
                        content = "\n".join(part.get("text", "") for part in content
                                            if isinstance(part, dict) and part.get("type") == "text")
                    if not isinstance(content, str):
                        content = ""
                    tool_calls = message.get("tool_calls") or []
                    tools_text = "\n".join(f"{call.get('function', {}).get('name')}: {call.get('function', {}).get('arguments')}"
                                           for call in tool_calls if isinstance(call, dict))
                    text = (content + ("\n" if content and tools_text else "") + tools_text).strip()
                    if text:
                        conversation.append({"role": role, "text": text[:5000]})
            video = result.get("video")
            video_path = Path(video) if video else None
            if video_path and video_path.is_file():
                video = video_path.resolve().relative_to(root.resolve()).as_posix() if video_path.resolve().is_relative_to(root.resolve()) else None
            else:
                video = None
            episodes.append({"model": protocol.get("model", run.get("model")),
                "effort": protocol.get("reasoning_effort", run.get("effort")),
                "task_id": protocol.get("task_id", run.get("task_id")),
                "task_name": protocol.get("task_name", LIBERO_10_TASKS[run["task_id"]][0]),
                "instruction": protocol.get("instruction", LIBERO_10_TASKS[run["task_id"]][1]),
                "init_state": state_id, "status": result.get("status",
                    ('running' if episode == max([int(e.get('episode', 0)) for e in events] + [len(by_episode)]) else 'pending')
                    if run.get('status') == 'running' else ('pending' if run.get('status') == 'pending' else 'missing_result')),
                "success": result.get("environment_success"),
                "reviewed_success": review.get("success"), "review_note": review.get("note"),
                "steps": result.get("steps"),
                "api_attempts": result.get("api_attempts", len(calls)),
                "estimated_usd": result.get("estimated_usd"),
                "budget_charge_usd": result.get("budget_charge_usd"),
                "stop_detail": result.get("stop_detail"), "framework_status": result.get("framework_status"),
                "video": video, "calls": calls, "trajectory": trajectory,
                "conversation": conversation, "protocol": protocol})
    return {"manifest": manifest, "episodes": episodes}


def write_report(root: Path) -> None:
    root = Path(root)
    data = collect(root)
    # JSON inside a script element must not permit an embedded closing tag.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (root / "report.html").write_text(HTML.replace("__DATA__", payload))
    public = {"manifest": data["manifest"], "episodes": [{k: v for k, v in e.items() if k != "calls"} for e in data["episodes"]]}
    (root / "results.json").write_text(json.dumps(public, indent=2))
    with (root / "episodes.csv").open("w", newline="") as stream:
        fields = ["model", "effort", "task_id", "init_state", "status", "success", "reviewed_success", "review_note", "steps", "api_attempts", "estimated_usd", "budget_charge_usd", "stop_detail"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["episodes"])


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Muse Glimmer · LIBERO experiments</title>
<style>
:root{font-family:system-ui,sans-serif;color:#edf3fa;background:#0b1420}*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:32px}h1{margin:0;font-size:36px}h2{font-size:21px;margin:0 0 14px}p,small{color:#b7c5d3}a{color:#80bfff}.top{border-bottom:1px solid #344657;padding-bottom:22px}.cards,.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:20px 0}.grid{grid-template-columns:repeat(2,1fr)}.card,.panel{background:#142333;border:1px solid #32485e;border-radius:12px;padding:18px;min-width:0}.card strong{display:block;font-size:29px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:18px 0}label{color:#b7c5d3}select{display:block;background:#20364b;color:#edf3fa;border:1px solid #52708a;border-radius:6px;padding:8px;min-width:170px}.inspector-head{display:flex;justify-content:space-between;gap:16px;align-items:end;flex-wrap:wrap}.inspector-head label{min-width:min(100%,420px)}#rollout{width:100%}.rollout-stats{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.rollout-stats span{background:#20364b;border-radius:6px;padding:8px 10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #304357;white-space:nowrap}th{color:#b7c5d3}.scroll{overflow:auto}.state-results{margin:20px 0}.state-results .scroll{max-height:500px}.state-results tr.pick:hover{background:#25425d}tr.pick{cursor:pointer}tr.pick:hover{background:#25425d}tr.pick[aria-selected="true"]{background:#294a5e}.ok{color:#70dab5}.bad{color:#ffb58c}video{width:100%;background:#050a10;border-radius:8px}pre{white-space:pre-wrap;overflow:auto;max-height:360px;background:#0b1825;padding:12px;border-radius:6px;font-size:12px}details{margin-top:15px}button{background:#254664;color:#edf3fa;border:1px solid #6182a1;padding:8px 12px;border-radius:6px;cursor:pointer}#plot{width:100%;min-height:130px}svg text{fill:#b7c5d3;font:12px system-ui}@media(max-width:950px){.grid,.cards{grid-template-columns:1fr 1fr}}@media(max-width:600px){.grid,.cards{grid-template-columns:1fr}main{padding:15px}}
</style></head><body><main><header class="top"><small>LIBERO-10 · local Muse Glimmer inference · local report</small><h1>Model rollout results</h1><p>Official success uses the simulator predicate. Reviewed success is recorded separately. Missing and interrupted episodes are shown separately from failures. This report reads saved experiment data and local videos.</p></header>
<div class="cards" id="cards"></div><div class="toolbar"><label>Model<select id="model"></select></label><label>Task number<select id="task"></select></label><label>Reasoning<select id="effort"></select></label><label>Initial state<select id="state"></select></label><button id="clear">Clear filters</button><a href="episodes.csv" download>Download episodes CSV</a><a href="results.json" download>Download results JSON</a></div>
<section class="panel state-results"><h2>Individual states</h2><p>Reviewed outcomes appear in parentheses after the official status.</p><p id="stateCount"></p><div class="scroll" id="episodes"></div></section>
<div class="grid"><section class="panel"><h2>Success by task</h2><div id="taskstats" class="scroll"></div></section><section class="panel"><h2>Model × reasoning</h2><div id="modelstats" class="scroll"></div></section></div>
<section class="panel" id="inspector" style="margin-bottom:20px"><div class="inspector-head"><div><h2>Rollout inspector</h2><div id="selected">Select a rollout</div></div></div><div id="episodeMeta"></div><div class="rollout-stats" id="rolloutStats"></div><div class="grid"><div><video id="video" controls preload="metadata"></video><p>Select a row in Individual states above to inspect its rollout. Agent view (left) and wrist view (right), 20 fps simulation time.</p><h2>Commanded end-effector path</h2><div id="trajectory"></div></div><div><h2>Local model calls and token use</h2><div id="plot"></div><div class="scroll" id="calls"></div></div></div><details><summary>Conversation and tool feedback</summary><pre id="conversation"></pre></details><details><summary>Protocol for this episode</summary><pre id="protocol"></pre></details></section>
<footer><p>Success is determined by the LIBERO simulator. Pending episodes are excluded from success rates. Early stops and errors are shown separately. Inference runs locally.</p></footer></main>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),E=D.episodes,$=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])),usd=v=>v==null?'—':'$'+Number(v).toFixed(3),num=v=>v==null?'—':Number(v).toLocaleString();
function options(id,values,allLabel,format=x=>x){$(id).innerHTML='<option value="">'+allLabel+'</option>'+values.map(v=>'<option value="'+esc(v)+'">'+esc(format(v))+'</option>').join('')}
options('model',[...new Set(E.map(e=>e.model).filter(Boolean))],'All models');options('task',[...new Set(E.map(e=>e.task_id).filter(v=>v!=null))].sort((a,b)=>a-b),'All tasks',v=>'Task '+v);options('effort',[...new Set(E.map(e=>e.effort).filter(Boolean))],'All reasoning');options('state',[...new Set(E.map(e=>e.init_state).filter(v=>v!=null))].sort((a,b)=>a-b),'All states',v=>'State '+v);
function table(headers,rows){return '<table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.join('')+'</tbody></table>'}
function cells(row){return '<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>'}
function filtered(){return E.filter(e=>(!$('model').value||e.model===$('model').value)&&(!$('effort').value||e.effort===$('effort').value)&&(!$('task').value||String(e.task_id)===$('task').value)&&(!$('state').value||String(e.init_state)===$('state').value))}
function attempted(e){return ['success','failure','incomplete_budget','api_error_usage_unknown','error','usage_unknown'].includes(e.status)}
function reviewedStatus(e){return e.status+(e.reviewed_success===true?' (success)':e.reviewed_success===false?' (failure)':'')}
function stats(rows,key){let groups=new Map();for(let e of rows){let name=key(e),g=groups.get(name)||{attempted:0,success:0,stopped:0,pending:0,calls:0};if(attempted(e)){g.attempted++;if(e.success)g.success++;if(e.status!=='success'&&e.status!=='failure')g.stopped++}else g.pending++;g.calls+=Number(e.api_attempts)||0;groups.set(name,g)}return [...groups].sort((a,b)=>String(a[0]).localeCompare(String(b[0]),undefined,{numeric:true})).map(([name,g])=>cells([name,g.success+'/'+g.attempted,g.attempted?((100*g.success/g.attempted).toFixed(1)+'%'):'—',g.stopped,g.pending,g.calls]))}
let chosen=null;
function draw(){let rows=filtered().sort((a,b)=>a.task_id-b.task_id||a.init_state-b.init_state||String(a.model).localeCompare(String(b.model))),done=rows.filter(attempted),success=done.filter(e=>e.success).length,pending=rows.length-done.length,first=rows.filter(e=>e.init_state===0&&attempted(e)),firstSuccess=first.filter(e=>e.success).length;$('cards').innerHTML=[['Scheduled',rows.length],['Success / attempted',success+' / '+done.length],['State 0 first pass',firstSuccess+' / '+first.length],['Unfinished / unknown',pending]].map(([k,v])=>'<div class="card"><small>'+esc(k)+'</small><strong>'+esc(v)+'</strong></div>').join('');$('taskstats').innerHTML=table(['Task','Success / attempted','Rate','Stopped early','Pending','Calls'],stats(rows,e=>'Task '+e.task_id));$('modelstats').innerHTML=table(['Model / effort','Success / attempted','Rate','Stopped early','Pending','Calls'],stats(rows,e=>e.model+' / '+e.effort));$('stateCount').textContent=rows.length+' state rollout'+(rows.length===1?'':'s')+' shown. Select a row to inspect its video.';$('episodes').innerHTML=table(['Model','Effort','Task','State','Status','Official success','Reviewed success','Steps','Calls'],rows.map(e=>'<tr class="pick" data-id="'+E.indexOf(e)+'">'+[e.model,e.effort,e.task_id,e.init_state,reviewedStatus(e),e.status==='success'?'Yes':attempted(e)?'No (stopped)':'Pending',e.reviewed_success===true?'Yes':e.reviewed_success===false?'No':'—',num(e.steps),num(e.api_attempts)].map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>'));document.querySelectorAll('tr.pick').forEach(tr=>tr.onclick=()=>{inspect(E[Number(tr.dataset.id)]);$('inspector').scrollIntoView({behavior:'smooth',block:'start'})});inspect(rows.includes(chosen)?chosen:rows[0]||null);}
function inspect(e){chosen=e;if(!e){$('selected').textContent='No rollout matches these filters';$('episodeMeta').textContent='';$('rolloutStats').innerHTML='';$('video').removeAttribute('src');$('video').style.display='none';$('plot').innerHTML='';$('calls').innerHTML='';$('conversation').textContent='';$('trajectory').textContent='';$('protocol').textContent='';return}document.querySelectorAll('tr.pick').forEach(tr=>tr.setAttribute('aria-selected',String(Number(tr.dataset.id)===E.indexOf(e))));$('rolloutStats').innerHTML=[['Official outcome',e.success?'Success':attempted(e)?e.status:'Pending'],['Reviewed success',e.reviewed_success===true?'Yes':e.reviewed_success===false?'No':'—'],['Simulator steps',num(e.steps)],['Local model calls',num(e.api_attempts)]].map(([k,v])=>'<span><small>'+esc(k)+'</small><br><strong>'+esc(v)+'</strong></span>').join('');$('selected').textContent=e.model+' / '+e.effort+' · task '+e.task_id+' · state '+e.init_state;$('episodeMeta').textContent=e.instruction+' · '+e.status+(e.stop_detail?' · '+e.stop_detail:'')+(e.review_note?' · Review: '+e.review_note:'');if(e.video){$('video').src=encodeURI(e.video);$('video').style.display='block'}else{$('video').removeAttribute('src');$('video').style.display='none'}let cs=e.calls||[],mx=Math.max(1,...cs.map(c=>Number(c.output_tokens)||0));$('plot').innerHTML='<svg viewBox="0 0 700 130" role="img" aria-label="Output tokens by model call">'+cs.map((c,i)=>'<rect x="'+(25+i*650/Math.max(1,cs.length))+'" y="'+(105-(Number(c.output_tokens)||0)/mx*85)+'" width="'+Math.max(2,650/Math.max(1,cs.length)-2)+'" height="'+((Number(c.output_tokens)||0)/mx*85)+'" fill="#70dab5"><title>Call '+c.call+': '+num(c.output_tokens)+' output tokens</title></rect>').join('')+'<text x="25" y="125">Call 1</text><text x="610" y="125">Call '+cs.length+'</text></svg>';$('calls').innerHTML=table(['Call','Input','Cached','Output','Reasoning','Latency','Tools','Status'],cs.map(c=>cells([c.call,num(c.input_tokens),num(c.cached_tokens),num(c.output_tokens),num(c.reasoning_tokens),c.latency_s==null?'—':Number(c.latency_s).toFixed(1)+' s',(c.tools||[]).join(', '),c.truncated?'Truncated':c.usage_unknown?'Usage unknown':c.http_status??'—'])));$('conversation').textContent=(cs.length?'Latest model response / tool call:\n'+(cs[cs.length-1].response_text||'Waiting for response')+'\n\n':'')+(e.conversation||[]).map(m=>m.role.toUpperCase()+'\n'+m.text).join('\n\n');trajectory(e.trajectory||[]);$('protocol').textContent=JSON.stringify(e.protocol,null,2)}
function trajectory(points){if(!points.length){$('trajectory').textContent='No action path recorded yet.';return}let coords=points.map(p=>[p[1],p[2]]),xs=coords.map(p=>p[0]),ys=coords.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),sx=x=>25+(x-xmin)/Math.max(.001,xmax-xmin)*600,sy=y=>145-(y-ymin)/Math.max(.001,ymax-ymin)*120;$('trajectory').innerHTML='<svg viewBox="0 0 650 170" role="img" aria-label="Commanded end-effector x-y path"><polyline fill="none" stroke="#70dab5" stroke-width="2" points="'+coords.map(p=>sx(p[0])+','+sy(p[1])).join(' ')+'"/><circle cx="'+sx(xs[0])+'" cy="'+sy(ys[0])+'" r="4" fill="#80bfff"/><circle cx="'+sx(xs.at(-1))+'" cy="'+sy(ys.at(-1))+'" r="4" fill="#ffb58c"/><text x="15" y="165">Start (blue) · End (orange) · x/y meters</text></svg>'}
for(let id of ['model','task','effort','state'])$(id).onchange=draw;$('clear').onclick=()=>{for(let id of ['model','task','effort','state'])$(id).value='';chosen=null;draw()};draw();
let loadedRevision=null;setInterval(async()=>{try{const response=await fetch('/revision',{cache:'no-store'});if(!response.ok)return;const changed=await response.text();if(loadedRevision!==null&&changed!==loadedRevision){location.reload()}loadedRevision=changed}catch{}},5000);
</script></body></html>'''

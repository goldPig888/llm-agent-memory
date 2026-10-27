"""Small read-only comparison for the sequential general harness experiments."""
import html
import json
import os
import time


def collect_ablations(root):
    campaigns=[]
    for path in sorted(root.glob('*-ablation-*.json')):
        try:data=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if not isinstance(data.get('protocol',{}).get('switches'),dict):continue
        trials={r['variant']:r for r in data.get('trials',[])}
        rows=[]
        for variant in data['protocol']['variants']:
            trial=dict(trials.get(variant,{}));trial['variant']=variant
            trial.setdefault('status','queued')
            if trial['status']=='running':
                try:os.kill(trial['pid'],0)
                except (OSError,KeyError):trial['status']='process missing'
                if time.time()-trial.get('checked_at',0)>45:trial['status']='stale status; inspect queue'
            rows.append(trial)
        campaigns.append(dict(name=path.stem,protocol=data['protocol'],trials=rows))
    return dict(campaigns=campaigns)


def render_ablations(data):
    esc=lambda x:html.escape(str(x))
    parts=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15">',
        '<title>General harness experiments</title><style>body{font:16px system-ui;background:#121722;color:#e1e7ef;margin:32px}a{color:#8ccaff}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #445;text-align:left}p{max-width:1000px}</style>',
        '<h1>General harness experiments</h1><a href="/report.html">All harness trials and videos</a>',
        '<p>Sequential Task 0/state 0 screen. Original general prompt, '
        '16,384 response tokens, 80 decisions, 900 physics steps. One episode per '
        'variant does not establish a success rate. A grasp proposal or completed '
        'primitive is not proof of a retained object or task success. Refreshes every 15 seconds.</p>']
    for campaign in data['campaigns']:
        parts+=['<h2>'+esc(campaign['name'])+'</h2>',
            '<p>Reasoning effort: <strong>'+esc(campaign['protocol'].get('effort','unknown'))+'</strong></p>',
            '<table><tr><th>Variant</th><th>Status</th><th>Decisions</th><th>Physics steps</th><th>First grasp proposed / completed (decision)</th><th>IK / collision / repeat rejections</th><th>Task success</th></tr>']
        for row in campaign['trials']:
            m=row.get('metrics',{});s=m.get('statuses',{})
            outcome=('yes' if m.get('environment_success') else 'no') if m.get('results') else 'pending'
            cells=[row['variant'],row['status'],m.get('decisions','—'),m.get('steps','—'),
                f"{m.get('first_grasp_proposed_call') or '—'} / {m.get('first_grasp_primitive_completed_call') or '—'}",
                f"{s.get('IK_FAILURE',0)} / {s.get('COLLISION',0)} / {s.get('repeat_rejected',0)}",outcome]
            parts.append('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in cells)+'</tr>')
        parts.append('</table>')
    if not data['campaigns']:parts.append('<p>No ablation campaign found.</p>')
    return '\n'.join(parts).encode()

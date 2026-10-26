"""Local, read-only viewer for pointing results and scoring overlays."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs,urlsplit


def page(dataset,results,selected):
    public=[json.loads(line) for line in (dataset/'samples.jsonl').read_text().splitlines()]
    summary=json.loads((results/'summary.json').read_text()) if (results/'summary.json').exists() else {'results':[],'groups':{}}
    by_id={r['id']:r for r in summary['results']}
    labels=json.loads((dataset/'scoring/labels.json').read_text())
    sample=next((s for s in public if s['id']==selected),public[0]);result=by_id.get(sample['id'])
    overlay=''
    if result and result.get('prediction'):
        u,v=result['prediction'];overlay+=f'<i style="left:{100*u}%;top:{100*v}%;background:#ff6b6b" title="Model prediction"></i>'
    center=labels[sample['id']].get('projected_center_px')
    if center:
        overlay+=f'<i style="left:{100*center[0]/511}%;top:{100*center[1]/511}%;background:#34d3ee" title="Projected body center: scoring only"></i>'
    rows=[]
    for s in public:
        r=by_id.get(s['id']);score=r.get('score',{}) if r else {}
        status='Pending' if not r else 'Parse error' if r.get('parse_error') else 'Hit' if score.get('hit_visible_object') else 'Not visible geometrically' if not score.get('target_visible') else 'Miss / abstention'
        error=score.get('pixel_error');err='—' if error is None else f'{error:.1f}'
        rows.append(f'<tr><td><a href="/?sample={escape(s["id"])}">{escape(s["id"])}</a></td><td>{status}</td><td>{err}</td></tr>')
    groups=''.join(f'<p>{escape(k)}: {v["hits"]}/{v["visible"]} visible-object hits; {v["parse_errors"]} parse errors.</p>' for k,v in summary.get('groups',{}).items())
    return f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>Glimmer pointing diagnostic</title><style>body{{font:16px system-ui;background:#0b1420;color:#edf3fa;margin:32px}}a{{color:#80bfff}}td,th{{padding:6px 18px;text-align:left}}.image{{position:relative;width:512px;max-width:100%}}img{{width:100%;display:block}}i{{position:absolute;width:10px;height:10px;border:1px solid white;border-radius:50%;transform:translate(-50%,-50%)}}</style>
<h1>Glimmer pointing diagnostic</h1><p>{len(summary['results'])}/{len(public)} samples recorded. No scored robot motion.</p>{groups}
<p>Red: model point. Cyan: projected object body center, used for scoring only. A visible object surface does not guarantee its label or identity is readable.</p>
<h2>{escape(sample['id'])}</h2><div class="image"><img src="/{escape(sample['image'])}" alt="Raw model input">{overlay}</div>
<table><thead><tr><th>Sample</th><th>Visible-mask result</th><th>Center error, pixels</th></tr></thead><tbody>{''.join(rows)}</tbody></table>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--results',required=True,type=Path)
    p.add_argument('--port',type=int,default=8769);args=p.parse_args()
    dataset=args.dataset.resolve();results=args.results.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request=urlsplit(self.path)
            if request.path in ('/','/report.html'):
                selected=parse_qs(request.query).get('sample',[''])[0]
                data=page(dataset,results,selected).encode();mime='text/html; charset=utf-8'
            else:
                path=(dataset/request.path.lstrip('/')).resolve()
                if path.parent!=dataset/'images' or path.suffix!='.png' or not path.is_file():
                    self.send_error(404);return
                data=path.read_bytes();mime='image/png'
            self.send_response(200);self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
    print(f'Pointing viewer: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()

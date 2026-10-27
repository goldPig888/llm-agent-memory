"""Read-only live comparison of xhigh native-box and normalized-point outputs."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


def page(dataset,results):
    progress=json.loads((results/'progress.json').read_text()) if (results/'progress.json').exists() else {'phase':'starting'}
    protocol=json.loads((results/'protocol.json').read_text()) if (results/'protocol.json').exists() else {}
    summary=json.loads((results/'summary.json').read_text()) if (results/'summary.json').exists() else {'groups':{},'results':[]}
    public={s['id']:s for s in map(json.loads,(dataset/'samples.jsonl').read_text().splitlines())}
    parts=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15">',
        '<title>Native format pilot — Glimmer xhigh</title><style>body{font:16px system-ui;background:#101823;color:#eef;margin:30px}a{color:#8cf}img{width:100%}.image{position:relative;width:512px}i{position:absolute;width:10px;height:10px;border-radius:50%;background:#f77;transform:translate(-50%,-50%)}.bbox{position:absolute;border:2px solid #6f9;box-sizing:border-box}pre{white-space:pre-wrap;max-width:1000px}article{margin:35px 0}</style>',
        '<h1>Native format pilot — Glimmer xhigh</h1>',
        '<p>Phase: '+escape(progress['phase'])+'; '+str(len(summary['results']))+'/'+str(protocol.get('samples',10))+' calls recorded.</p>',
        '<p>Five saved agent-camera states, paired prompts, fresh context, xhigh, 16,384-token cap. '
        'Box centers use the predeclared 0–1000 scale assumption; the prompt does not specify a scale. '
        'Raw outputs appear below. No robot motion or fine-tuning. Historical 44% used high effort and 50 states.</p>']
    for name,group in summary['groups'].items():
        parts.append('<p>'+escape(name)+': '+str(group['hits'])+'/'+str(group['visible'])+' target-mask hits; '+str(group['parse_errors'])+' parse errors.</p>')
    for row in summary['results']:
        sample=public[row['id']];overlay=''
        if row.get('prediction'):
            u,v=row['prediction'];overlay+=f'<i style="left:{100*u}%;top:{100*v}%"></i>'
        if row.get('bbox'):
            b=row['bbox'];overlay+=f'<div class="bbox" style="left:{b["x_min"]/10}%;top:{b["y_min"]/10}%;width:{(b["x_max"]-b["x_min"])/10}%;height:{(b["y_max"]-b["y_min"])/10}%"></div>'
        score=row['score'];outcome=row.get('parse_error') or ('Hit' if score['hit_visible_object'] else 'Miss / abstention')
        parts.append('<article><h2>'+escape(row['id']+' / '+row['model'])+'</h2><p>'+escape(outcome)+'</p>'
            '<div class="image"><img src="/'+escape(sample['image'])+'" alt="Unmodified model input">'+overlay+'</div>'
            '<pre>'+escape(row.get('raw_output',''))+'</pre></article>')
    return '\n'.join(parts).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--port',type=int,default=8770);args=p.parse_args()
    dataset=args.dataset.resolve();results=args.results.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ('/','/report.html'):data=page(dataset,results);mime='text/html; charset=utf-8'
            else:
                target=(dataset/path.lstrip('/')).resolve()
                if target.parent!=dataset/'images' or target.suffix!='.png' or not target.is_file():
                    self.send_error(404);return
                data=target.read_bytes();mime='image/png'
            self.send_response(200);self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()

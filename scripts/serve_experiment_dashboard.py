"""Serve the local experiment dashboard, refreshing it when opened in a browser."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

from experiment_report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = args.output.resolve()
    if not (root / 'campaign.json').exists():
        parser.error('campaign.json is missing')
    lock = threading.Lock()

    def refresh_if_stale() -> None:
        """Rebuild only when the manifest is newer than the rendered report."""
        manifest = root / 'campaign.json'
        report = root / 'report.html'
        if not report.exists() or manifest.stat().st_mtime_ns > report.stat().st_mtime_ns:
            write_report(root)

    class Handler(SimpleHTTPRequestHandler):
        def _allowed(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                path = '/report.html'
            relative = Path(path.lstrip('/'))
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                return None
            if path not in {'/report.html', '/episodes.csv', '/results.json', '/campaign.json'} and target.name != 'rollout.mp4':
                return None
            return path

        def end_headers(self):
            if self._allowed() in {'/report.html', '/episodes.csv', '/results.json', '/campaign.json'}:
                self.send_header('Cache-Control', 'no-store, max-age=0')
            super().end_headers()

        def do_GET(self):
            path = self._allowed()
            if path is None:
                self.send_error(404)
                return
            if path == '/report.html':
                with lock:
                    try:
                        refresh_if_stale()
                    except Exception as exc:
                        self.log_error('refresh failed; serving last report: %s', type(exc).__name__)
            self.path = path
            super().do_GET()

        def do_HEAD(self):
            path = self._allowed()
            if path is None:
                self.send_error(404)
                return
            self.path = path
            super().do_HEAD()

    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(root)))
    print(f'Dashboard: http://{args.host}:{args.port}/report.html', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

"""Export current experiment numbers to one JSON file and upload it to Drive."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

from fetch_rollout_videos import DEFAULT_SOURCE
from upload_rollout_videos import DEFAULT_FOLDER


def read_json(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def aggregate(rows):
    attempted = [row for row in rows if row['has_result']]
    known = [row for row in attempted if isinstance(row.get('success'), bool)]
    successes = sum(row.get('success') is True for row in known)
    return {'scheduled': len(rows), 'attempted': len(attempted),
            'successes': successes, 'failures': len(known) - successes,
            'unknown_outcomes': len(attempted) - len(known),
            'pending_or_missing': len(rows) - len(attempted),
            'success_rate': successes / len(known) if known else None,
            **{field: sum(row.get(field) or 0 for row in rows) for field in
               ('steps', 'api_attempts', 'estimated_usd', 'budget_charge_usd', 'input_tokens',
                'cached_tokens', 'output_tokens', 'reasoning_tokens', 'latency_s')}}


def export_results(source, visualization_data=False, video_index=None, manifest=None):
    manifest = read_json(source / 'campaign.json', {}) if manifest is None else manifest
    annotations = read_json(source / 'manual_annotations.json', {}).get('episodes', [])
    reviews = {(row['directory'], row['init_state']): row for row in annotations}
    episodes = []
    for run in manifest.get('runs', []):
        if run.get('status') == 'superseded':
            continue
        folder = (source / run['directory']).resolve()
        if not folder.is_relative_to(source.resolve()):
            raise ValueError('Run directory is outside the campaign')
        summary = read_json(folder / 'summary.json', {})
        protocol = read_json(folder / 'protocol.json', summary.get('protocol', {}))
        results = {row['episode']: row for row in summary.get('results', [])}
        calls = defaultdict(dict)
        usage_files = [folder / 'api' / 'api_usage.jsonl']
        if protocol.get('resume_from'):
            previous = Path(protocol['resume_from']).resolve()
            if not previous.is_relative_to(source.resolve()):
                raise ValueError('Continuation source is outside the campaign')
            usage_files.insert(0, previous / 'api/api_usage.jsonl')
        for usage_file in usage_files:
            if not usage_file.is_file():
                continue
            with usage_file.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('event') in {'response', 'usage_unknown'}:
                        calls[event.get('episode')][event.get('call')] = event
        for episode, state in enumerate(protocol.get('initial_state_ids', run.get('initial_state_ids', []))):
            if state in run.get('superseded_initial_state_ids', []):
                continue
            result = results.get(episode, {})
            review = reviews.get((run['directory'], state), {})
            row = {'model': protocol.get('model', run.get('model')),
                   'effort': protocol.get('reasoning_effort', run.get('effort')),
                   'task_id': protocol.get('task_id', run.get('task_id')),
                   'init_state': state, 'episode': episode, 'run': run['directory'],
                   'status': result.get('status', 'pending' if run.get('status') in {'pending', 'running'} else 'missing_result'),
                   'has_result': episode in results, 'success': result.get('environment_success'),
                   'reviewed_success': review.get('success'),
                   'stop_detail': result.get('stop_detail'),
                   'metrics': result.get('metrics', {})}
            if row['status'] == 'incomplete_budget' and not row['stop_detail'] and result.get('api_attempts', 0) >= protocol.get('max_api_attempts_per_episode', run.get('max_calls', 80)):
                row.update(status='failure', stop_detail='call_limit')
            for field in ('steps', 'api_attempts', 'estimated_usd', 'budget_charge_usd'):
                row[field] = result.get(field)
            row['calls'] = []
            for call, event in sorted(calls[episode].items(), key=lambda item: item[0] or 0):
                usage = event.get('usage') or {}
                cached = usage.get('input_tokens_details') or {}
                reasoning = usage.get('output_tokens_details') or usage.get('thinking_tokens') or {}
                total_input = usage.get('input_tokens')
                if protocol.get('wire') == 'messages' and total_input is not None:
                    total_input += (usage.get('cache_creation_input_tokens') or 0) + (usage.get('cache_read_input_tokens') or 0)
                record = {'call': call, 'input_tokens': total_input,
                          'cached_tokens': cached.get('cached_tokens', usage.get('cache_read_input_tokens')),
                          'output_tokens': usage.get('output_tokens'),
                          'reasoning_tokens': reasoning.get('reasoning_tokens', reasoning.get('thinking_tokens')) if isinstance(reasoning, dict) else reasoning,
                          'estimated_usd': event.get('estimated_usd'), 'latency_s': event.get('latency_s'),
                          'http_status': event.get('http_status'), 'usage_unknown': event['event'] == 'usage_unknown'}
                row['calls'].append(record)
            for field in ('input_tokens', 'cached_tokens', 'output_tokens', 'reasoning_tokens', 'latency_s'):
                values = [call[field] for call in row['calls'] if isinstance(call[field], (int, float))]
                row[field] = sum(values) if values else None
            episodes.append(row)
    if visualization_data or video_index:
        if visualization_data:
            from response_visualization import visualization
        links = read_json(video_index, {'videos': []})['videos'] if video_index else []
        for row in episodes:
            if visualization_data:
                row['trajectory'] = visualization(source / row['run'], row['episode'], row['init_state'], row['calls'])
                from export_hindsight import attach_hindsight
                attach_hindsight(row)
            candidates = [link for link in links if link['source_relative'].startswith(row['run'] + f'/episode_{row["episode"]:02d}/')]
            if len(candidates) == 1:
                row['video'] = {key: candidates[0].get(key) for key in ('drive_file_id', 'preview_url', 'view_url')}
    groups = defaultdict(list)
    for row in episodes:
        groups[(row['model'], row['effort'], row['task_id'])].append(row)
    return {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'campaign': source.name, 'totals': aggregate(episodes),
            'by_model_task': [dict(model=model, effort=effort, task_id=task, **aggregate(rows))
                              for (model, effort, task), rows in sorted(groups.items())],
            'episodes': episodes}


def remove_superseded_results(existing, manifest):
    """Remove archived local attempts while retaining other uploaded sources."""
    runs = {row['directory']: row for row in manifest.get('runs', [])}
    retained = []
    for episode in existing.get('episodes', []):
        run = runs.get(episode.get('run'))
        matches = run is not None and all(
            episode.get(field) == run.get(field)
            for field in ('model', 'effort', 'task_id'))
        if matches and (run.get('status') == 'superseded' or
                        episode.get('init_state') in run.get('superseded_initial_state_ids', [])):
            continue
        retained.append(episode)
    return dict(existing, episodes=retained)


def merge_results(existing, current):
    """Append new episode records while making repeated uploads idempotent."""
    if current.get('replace_episode_outcomes'):
        keys = {tuple(row.get(field) for field in ('model', 'effort', 'task_id', 'init_state'))
                for row in current.get('episodes', [])}
        existing = dict(existing, episodes=[row for row in existing.get('episodes', [])
                        if tuple(row.get(field) for field in ('model', 'effort', 'task_id', 'init_state')) not in keys])
    episodes = {}
    for row in [*existing.get('episodes', []), *current.get('episodes', [])]:
        key = tuple(row.get(field) for field in
                    ('model', 'effort', 'task_id', 'init_state', 'run', 'episode'))
        episodes[key] = row
    rows = list(episodes.values())
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['effort'], row['task_id'])].append(row)
    merged = dict(current)
    merged['episodes'] = rows
    merged['totals'] = aggregate(rows)
    merged['by_model_task'] = [
        dict(model=model, effort=effort, task_id=task, **aggregate(group))
        for (model, effort, task), group in sorted(groups.items())]
    merged['merged_sources'] = sorted(set(existing.get('merged_sources', [existing.get('campaign')]) +
                                          current.get('merged_sources', [current.get('campaign')])) - {None})
    return merged


def upload(archive, source, current, args):
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_authorized_user_file(str(args.token), scopes) if args.token.exists() else None
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not args.credentials:
                raise ValueError('--credentials is required for first-time OAuth login')
            credentials = InstalledAppFlow.from_client_secrets_file(str(args.credentials), scopes).run_local_server(port=0, open_browser=False)
        args.token.parent.mkdir(parents=True, exist_ok=True)
        args.token.touch(mode=0o600, exist_ok=True)
        args.token.chmod(0o600)
        args.token.write_text(credentials.to_json())
    transport = httplib2.Http(timeout=60)
    # Drive uses HTTP 308 to acknowledge resumable chunks, not as a redirect.
    transport.redirect_codes = transport.redirect_codes - {308}
    service = build('drive', 'v3', http=AuthorizedHttp(credentials, http=transport))
    folder = service.files().get(fileId=args.folder_id, fields='mimeType,capabilities(canAddChildren)', supportsAllDrives=True).execute()
    if folder['mimeType'] != 'application/vnd.google-apps.folder' or not folder.get('capabilities', {}).get('canAddChildren'):
        raise ValueError('Destination must be a folder you can upload into')
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    if args.file_id:
        found = [service.files().get(fileId=args.file_id, fields='id,name,parents',
                                     supportsAllDrives=True).execute()]
        if args.folder_id not in found[0].get('parents', []):
            raise ValueError('--file-id is not inside the selected Drive folder')
    else:
        queries = [
            f"'{args.folder_id}' in parents and trashed = false and appProperties has {{ key='experimentResults' and value='{key}' }}",
            f"'{args.folder_id}' in parents and trashed = false and name = '{archive.name}'",
        ]
        by_id = {}
        for query in queries:
            files = service.files().list(q=query, fields='files(id,name,parents)', supportsAllDrives=True,
                                         includeItemsFromAllDrives=True).execute()['files']
            by_id.update((item['id'], item) for item in files)
        found = list(by_id.values())
        if len(found) > 1:
            raise ValueError('Multiple result files match; select one with --file-id')
    if found:
        stream = io.BytesIO()
        downloader = MediaIoBaseDownload(
            stream, service.files().get_media(fileId=found[0]['id'], supportsAllDrives=True), chunksize=16 * 1024**2)
        done = False
        while not done:
            progress, done = downloader.next_chunk(num_retries=5)
            if progress:
                print(f'Reading existing results: {progress.progress():.1%}', flush=True)
        existing = json.loads(stream.getvalue())
        if existing.get('version') != 1 or not isinstance(existing.get('episodes'), list):
            raise ValueError('Existing Drive file is not a supported experiment results export')
        archive.with_suffix('.before-upload.json').write_bytes(stream.getvalue())
        existing = remove_superseded_results(existing, read_json(source / 'campaign.json', {}))
        current = merge_results(existing, current)
        archive.write_text(json.dumps(current, indent=2, allow_nan=False) + '\n')
    media = MediaFileUpload(str(archive), mimetype='application/json', chunksize=16 * 1024**2, resumable=True)
    if found:
        request = service.files().update(fileId=found[0]['id'], body={'name': archive.name},
                                         media_body=media, fields='id,webViewLink', supportsAllDrives=True)
    else:
        request = service.files().create(
            body={'name': archive.name, 'parents': [args.folder_id],
                  'appProperties': {'experimentResults': key}},
            media_body=media, fields='id,webViewLink', supportsAllDrives=True)
    response = None
    while response is None:
        progress, response = request.next_chunk(num_retries=5)
        if progress:
            print(f'Uploaded {progress.progress():.1%}', flush=True)
    file_id = response['id']
    receipt = {'file_id': file_id, 'drive_name': archive.name,
               'view_url': response.get('webViewLink', f'https://drive.google.com/file/d/{file_id}/view'),
               'download_url': f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media',
               'export': str(archive), 'created_at': current['created_at'],
               'episodes': len(current['episodes']), 'size_bytes': archive.stat().st_size}
    archive.with_suffix('.drive.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiment_results.json'))
    parser.add_argument('--folder-id', default=DEFAULT_FOLDER)
    parser.add_argument('--file-id', help='Existing results file to merge into; recommended for uploads from another machine')
    parser.add_argument('--token', type=Path, default=Path('outputs/drive_auth/token.json'))
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--visualization', action='store_true', help='Include response markers, tool targets, and interpolated action paths')
    parser.add_argument('--video-index', type=Path, default=Path('outputs/website_videos/drive_videos.json'))
    parser.add_argument('--export-only', action='store_true', help='Write JSON without uploading')
    parser.add_argument('--prepared-input', type=Path, help='Upload an already prepared export without regenerating it')
    parser.add_argument('--dry-run', action='store_true', help='Show counts without writing or uploading')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.folder_id):
        parser.error('Invalid Drive folder ID')
    if not (args.source / 'campaign.json').is_file():
        parser.error('Source must contain campaign.json')
    try:
        data = read_json(args.prepared_input, {}) if args.prepared_input else export_results(args.source, args.visualization, args.video_index)
        if data.get('version') != 1 or not isinstance(data.get('episodes'), list):
            raise ValueError('Input must be a supported experiment results export')
        payload = json.dumps(data, indent=2, allow_nan=False) + '\n'
        print(f'{len(data["episodes"])} episodes; {len(payload.encode()) / 1024:.1f} KiB JSON', flush=True)
        if args.dry_run:
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(payload)
        temporary.replace(args.output)
        print(f'Results: {args.output}', flush=True)
        if not args.export_only:
            upload(args.output, args.source, data, args)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

"""Expose stop-tool hindsight and export model-written post-mortems without reruns."""
import argparse
import csv
import json
from pathlib import Path


def attach_hindsight(episode):
    """Keep every stop call, including an explicit 'none', without interpreting it."""
    records = []
    for call in episode.get('calls', []):
        for tool in call.get('tools', []):
            if not isinstance(tool, dict) or tool.get('name') not in {'done', 'give_up'}:
                continue
            arguments = tool.get('arguments', {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, dict):
                continue
            hindsight = arguments.get('hindsight')
            if not isinstance(hindsight, str) or not hindsight.strip():
                continue
            call['hindsight'] = hindsight
            records.append({
                **{key: episode.get(key) for key in
                   ('model', 'effort', 'task_id', 'init_state', 'run', 'episode',
                    'status', 'success', 'steps', 'api_attempts', 'stop_detail')},
                'call': call.get('call'), 'stop_tool': tool['name'],
                'summary': arguments.get('summary'), 'reason': arguments.get('reason'),
                'hindsight': hindsight,
            })
    if records:
        episode['hindsight'] = records[-1]['hindsight']
        episode['hindsight_call'] = records[-1]['call']
        episode['hindsight_stop_tool'] = records[-1]['stop_tool']
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--enriched-output', type=Path,
                        help='Optionally write the full results with explicit hindsight fields')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    records = [record for episode in data['episodes'] for record in attach_hindsight(episode)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'version': 1, 'description': 'Verbatim model-written stop-tool hindsight; absence is not an inferred post-mortem.',
               'source_episodes': len(data['episodes']), 'records': records}
    (args.output_dir / 'hindsight.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    fields = ['model', 'effort', 'task_id', 'init_state', 'run', 'episode', 'status',
              'success', 'steps', 'api_attempts', 'stop_detail', 'call', 'stop_tool',
              'summary', 'reason', 'hindsight']
    with (args.output_dir / 'hindsight.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if args.enriched_output:
        args.enriched_output.parent.mkdir(parents=True, exist_ok=True)
        args.enriched_output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(f'Exported {len(records)} hindsight records from {len(data["episodes"])} episodes.')


if __name__ == '__main__':
    main()

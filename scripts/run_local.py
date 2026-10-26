"""Evaluate local Muse Glimmer on the paper's LIBERO-10 protocol."""
import argparse
import json
import hashlib
import importlib.metadata
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from inspect_robots import eval as evaluate
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.task import Task
from inspect_robots_agent.policy import agent_policy
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import LIBERO_10_TASKS, libero_10_task
from inspect_robots_libero.embodiment import BASE_DOCS


def local_url(value):
    u = urlsplit(value)
    if u.scheme != 'http' or u.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('Inference endpoint must be local HTTP loopback')
    return value.rstrip('/')


class Capture(httpx.BaseTransport):
    def __init__(self, folder, effort, inner=None, max_tokens=16384):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.effort = effort
        self.max_tokens = max_tokens
        self.episode = 0
        self.calls = 0
        self.truncations = 0
        self.http_errors = []
        self.inner = inner or httpx.HTTPTransport(trust_env=False)

    def event(self, **values):
        with (self.folder / 'api_usage.jsonl').open('a') as f:
            f.write(json.dumps(dict(episode=self.episode, call=self.calls, **values)) + '\n')

    def handle_request(self, request):
        local_url(str(request.url))  # No redirects or remote inference fallback.
        self.calls += 1
        body = json.loads(request.read())
        body.pop('reasoning_effort', None)
        body['chat_template_kwargs'] = {'reasoning_strength': self.effort}
        body['max_tokens'] = self.max_tokens
        prefix = self.folder / f'ep{self.episode:02d}_call{self.calls:02d}'
        prefix.with_name(prefix.name + '_request.json').write_text(json.dumps(body))
        self.event(event='request_started')
        start = time.monotonic()
        response = self.inner.handle_request(httpx.Request(
            request.method, request.url, json=body,
            headers={'Content-Type': 'application/json'}, extensions=request.extensions))
        response.read()
        prefix.with_name(prefix.name + '_response.json').write_bytes(response.content)
        payload = response.json()
        self.truncations += sum(c.get('finish_reason')=='length' for c in payload.get('choices',[]))
        if response.status_code >= 400:
            self.http_errors.append(dict(status=response.status_code,error=payload.get('error')))
        usage = payload.get('usage', {})
        self.event(event='response', http_status=response.status_code,
                   latency_s=time.monotonic() - start, estimated_usd=0,
                   usage={'input_tokens': usage.get('prompt_tokens'),
                          'output_tokens': usage.get('completion_tokens'),
                          'input_tokens_details': usage.get('prompt_tokens_details', {}),
                          'output_tokens_details': usage.get('completion_tokens_details', {})})
        return response

    def close(self):
        self.inner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/muse'))
    parser.add_argument('--tasks', type=int, nargs='+', choices=range(10), default=list(range(10)))
    parser.add_argument('--inits', type=int, nargs='+', choices=range(50), default=list(range(10)))
    parser.add_argument('--base-url', type=local_url, default='http://127.0.0.1:8081/v1')
    parser.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh'], default='medium')
    parser.add_argument('--max-calls', type=int, default=80)
    parser.add_argument('--max-steps', type=int, default=900)
    parser.add_argument('--image-horizon', type=int, default=2)
    parser.add_argument('--max-output-tokens', type=int, default=16384)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--run-name',help='Stable new run directory for one task')
    args = parser.parse_args()
    if min(args.max_calls, args.max_steps, args.image_horizon, args.max_output_tokens) < 1:
        parser.error('Limits must be positive')
    if args.run_name and (len(args.tasks)!=1 or Path(args.run_name).name!=args.run_name):
        parser.error('--run-name requires one task and a simple directory name')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'campaign.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'name': 'Muse Glimmer local LIBERO-10', 'runs': []}
    if not args.plan_only:
        with httpx.Client(trust_env=False, timeout=30) as client:
            client.get(args.base_url + '/models').raise_for_status()
            props=client.get(args.base_url.removesuffix('/v1')+'/props')
            props.raise_for_status()
            server=props.json()
    else:
        server={}
    for task_id in args.tasks:
        directory = args.run_name or f'task{task_id:02d}_{args.effort}_' + time.strftime('%Y%m%d-%H%M%S')
        folder = root / directory
        folder.mkdir(exist_ok=False)
        protocol = dict(model='muse-glimmer', wire='chat', task_id=task_id,
                        task_name=LIBERO_10_TASKS[task_id][0], instruction=LIBERO_10_TASKS[task_id][1],
                        initial_state_ids=args.inits, reasoning_effort=args.effort,
                        image_horizon=args.image_horizon, max_steps=args.max_steps,
                        max_api_attempts_per_episode=args.max_calls, max_output_tokens=args.max_output_tokens,
                        zero_shot=True, prior_learnings=None, local_inference=True,
                        base_url=args.base_url, quantization='official KQuant-17GB-Q4_K_M',
                        server_context_tokens=server.get('default_generation_settings',{}).get('n_ctx'),
                        server_generation_defaults=server.get('default_generation_settings',{}),
                        server_build=server.get('build_info'),
                        baseline_protocol=True,harness=False,demonstrations=False,task_notes=False,
                        env_seeds=[142+i for i in args.inits],evaluation_seeds=[42+i for i in args.inits],
                        image_resolution=[256,256],control_hz=20,
                        embodiment_docs=BASE_DOCS,
                        embodiment_docs_sha256=hashlib.sha256(BASE_DOCS.encode()).hexdigest(),
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        package_versions={name:importlib.metadata.version(name) for name in
                            ['hf-libero','mujoco','robosuite','inspect-robots','inspect-robots-agent']},
                        deviations_from_draft=['Local Chat API with quantized model',
                            '16384 output tokens rather than draft 3072'] if args.max_output_tokens==16384 else ['Local Chat API with quantized model'])
        (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        run = dict(directory=directory, model='muse-glimmer', effort=args.effort,
                   task_id=task_id, initial_state_ids=args.inits, status='pending' if args.plan_only else 'running')
        manifest['runs'].append(run)
        def save():
            (folder / 'summary.json').write_text(json.dumps(dict(protocol=protocol, results=results), indent=2))
            tmp = manifest_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(manifest, indent=2)); tmp.replace(manifest_path)
        results = []
        save()
        if args.plan_only:
            continue
        try:
            for ep, state in enumerate(args.inits):
                scene = libero_10_task(task_id, num_scenes=1, start=state, max_steps=args.max_steps).scenes[0]
                embodiment = LiberoEmbodiment(task_id=task_id, output_dir=folder / f'episode_{ep:02d}')
                capture = Capture(folder / 'api', args.effort, max_tokens=args.max_output_tokens); capture.episode = ep
                policy = agent_policy(model='muse-glimmer', base_url=args.base_url, wire='chat',
                    env={}, effort=None, depth='off',
                    max_llm_calls=args.max_calls, image_horizon=args.image_horizon,
                    prior_learnings=None, transport=capture, wire_capture=True)
                policy._client._http.timeout = httpx.Timeout(1800, connect=30)
                task = Task(name=f'libero_10_task{task_id:02d}', scenes=[scene],
                            scorer=libero_10_task(task_id).scorer, max_steps=args.max_steps)
                space = embodiment.info.action_space
                try:
                    (log,) = evaluate(task, policy, embodiment,
                        log_dir=str(folder / f'episode_{ep:02d}' / 'inspect'), seed=42 + state,
                        store_frames=True, store_actions=True, grader=None,
                        approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)))
                    last = embodiment.last or {}
                    success = bool(last.get('success'))
                    row = dict(episode=ep, init_state=state, status='success' if success else ('error' if log.status == 'error' else 'failure'),
                        environment_success=success, framework_status=log.status,
                        metrics=log.results.metrics if log.results else None,
                        steps=last.get('steps', 0), api_attempts=capture.calls, estimated_usd=0,
                        output_truncations=capture.truncations,api_errors=capture.http_errors,
                        video=str(embodiment.episode_dir / 'rollout.mp4') if embodiment.episode_dir else None)
                finally:
                    embodiment.close(); policy._client.close()
                results.append(row); save()
                print(json.dumps(row), flush=True)
            run['status'] = 'completed'
        except BaseException:
            run['status'] = 'interrupted'
            raise
        finally:
            save()


if __name__ == '__main__':
    main()

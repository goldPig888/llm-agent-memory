"""Open Gemini harness on one Panda in LIBERO, using local Glimmer or Haimaker DeepSeek."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from inspect_robots_libero.tasks import libero_10_task, LIBERO_10_TASKS
from llm_agent_memory.harness.session import Session
from llm_agent_memory.harness.upstream import SYSTEM_PROMPT, extract_curl
from run_local import Capture, local_url
from inspect_robots._dotenv import read_dotenv
from llm_agent_memory.harness.haimaker import BatchBudget,HaimakerCapture,ProviderStop,MODEL as DEEPSEEK_MODEL
from llm_agent_memory.harness.experiments import VARIANTS, RecoveryMonitor, image_instructions


def prompt(profile='wrist-v3'):
    text=SYSTEM_PROMPT.replace('Gemini 4 Argon controlling a real 6-DoF YAM robot arm',
                               'Muse Glimmer controlling a simulated 7-DoF Panda robot arm')
    text=text.replace('A human operator previews every motion segment in 3D and must approve it\nbefore the arm moves. If they reject, their feedback is in the response --\nfollow it.',
        'An automatic Panda IK and swept-contact checker previews each segment.\nRejected segments return IK_FAILURE or COLLISION; correct the proposal.')
    text=text.replace('Arm base frame: +x forward (away from the arm base), +y left, +z up.',
                      'LIBERO world frame: x and y follow the calibrated grid, +z up.')
    text=text.replace('Overhead camera (calibrated)', 'Fixed agentview camera (calibrated; oblique rather than overhead)')
    text=text.replace('from stereo depth (noisy, ~1-2 cm)', 'from the simulated camera depth sensor')
    if profile=='wrist-v3':
        text=text.replace('deproject through the overhead image only.',
            'choose camera="agentview" for coarse targeting or camera="wrist" for fine targeting.')
        text=text.replace('of the LAST overhead image','of the selected camera image')
        text=text.replace('approach_height=0.08','approach_height=0.05')
        text=text.replace('Stay inside obs.workspace.', 'Stay inside x,y in [-0.6,0.6], z in [table_z,1.6].')
        text+='''\n## Two-stage Panda targeting
Agentview is oblique; a wrong height plane shifts world x,y substantially.
Coarse: choose the requested object carefully, then use
x,y,z=robot.deproject_depth(u=...,v=...,camera="agentview")
and robot.goto(x=x,y=y,z=z+0.13,pitch_deg=90,yaw_deg=0).
Depth returns the visible surface, not an object's center. The first move is
only a hover: re-observe before descending. Confirm the actual wrist view is
top-down and the object is visible; do not assume this for a tilted hand.
Fine: click the object's top center in the full wrist image, then use
x,y=robot.deproject(u=...,v=...,z_plane=...,camera="wrist").
Each camera's normalized u,v are in its displayed image (top-left origin).
Only agentview is cropped by look_at; wrist stays full-frame.
Use the correct grasp height and bracket the object between the jaws.
robot.grasp accepts support_z: record the support surface when known.
After lift, visually verify object motion; holding alone is only a proxy.
robot.place_relative(x=...,y=...,support_z=...,release_clearance=0.01,
 yaw_deg=...,approach_height=0.05) uses the retained grasp's measured height
offset. The destination support height must be chosen from observation; it
does not infer basket floors, rim heights, or objects from hidden poses.
At most two consecutive zoom-only plans are allowed. Choose a target and
propose a hover or corrected action after that; repeated zooms waste calls.
Use steps_remaining to plan: contact descent is slow only in its last 2 cm.
'''
    start=text.index('- Reach:')
    end=text.index('- After a grasp',start)
    text=text[:start]+text[end:]
    text += '''\n## LIBERO adaptation\nThis is a single arm: only robot.<primitive> is valid. No left/right or parallel calls.
The shell command is only a response format: it is parsed, never executed.
At most 80 model decisions and 900 simulator steps are available. Each primitive
may consume many steps; the simulator checks task success after every step.
Gripper command is 0=open, 1=close; finger_positions are measured in meters.
holding is only a finger-gap estimate, not proof of an object grasp: verify visually
after lifting. Only the automatic LIBERO predicate establishes task success.
Use measured poses and tracking errors. Closed fingers on nothing are not success.
Grid is 2 cm; labels give world x,y in meters. Both images are upright 512x512.
Camera pixel coordinates for deproject are normalized to the current cropped
agentview image. Pure deprojection returns coordinates inside a plan; no motion.
The table_z and workspace bounds are in the observation, not zero-height coordinates.
Keep # Thought short (40 words). On turn one include # Plan and granularity.
Return one complete curl /obs or curl /plan with a complete heredoc. Do not return
JSON, markdown code fences, unrelated shell commands, or simulator internals.
'''
    if profile=='wrist-v3':
        text=text.replace('Camera pixel coordinates for deproject are normalized to the current cropped\nagentview image.',
            'Camera pixel coordinates are normalized to the selected displayed image. Agentview is cropped; wrist is full-frame.')
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/muse_harness_xhigh'))
    parser.add_argument('--append',action='store_true',help='Preserve earlier trials and append a new run')
    parser.add_argument('--trial-label',default='')
    parser.add_argument('--inits',type=int,nargs='+',default=list(range(10)),choices=range(50))
    parser.add_argument('--effort',choices=['low','medium','high','xhigh','max'],default='xhigh')
    parser.add_argument('--provider',choices=['local','haimaker'],default='local')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--api-budget',type=float,default=5.)
    parser.add_argument('--profile',choices=['legacy','wrist-v3'],default='wrist-v3')
    parser.add_argument('--variant',choices=list(VARIANTS),default='control')
    parser.add_argument('--generation-seed',type=int,default=None)
    parser.add_argument('--max-calls',type=int,default=80)
    parser.add_argument('--max-steps',type=int,default=900)
    parser.add_argument('--max-output-tokens',type=int,default=16384)
    parser.add_argument('--base-url',type=local_url,default='http://127.0.0.1:8081/v1')
    args=parser.parse_args()
    if min(args.max_calls,args.max_steps,args.max_output_tokens)<1: parser.error('limits must be positive')
    remote=args.provider=='haimaker'
    if remote and args.effort!='max': parser.error('DeepSeek trial requires --effort max')
    if not remote and args.effort=='max': parser.error('Local Glimmer highest setting is xhigh')
    if remote and not 0<args.api_budget<=5: parser.error('API batch budget must be >0 and <=5 USD')
    if remote and args.generation_seed is not None: parser.error('No generation seed supported for API trial')
    if remote:
        settings={**(read_dotenv(args.env_file) if args.env_file else {}),**os.environ}
        key=settings.get('HAIMAKER_API_KEY')
        if not key:parser.error('HAIMAKER_API_KEY is required')
    model=DEEPSEEK_MODEL if remote else 'muse-glimmer'
    base_url='https://api.haimaker.ai/v1' if remote else args.base_url
    variant=VARIANTS[args.variant]
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    trial_lock=(root/'.trial.lock').open('a')
    try:
        fcntl.flock(trial_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another trial is already running in this viewer campaign')
    manifest_path=root/'campaign.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        name='Open Gemini harness adaptation / Muse Glimmer',runs=[])
    if manifest.get('runs') and not args.append:
        parser.error('output already contains a campaign; use a fresh directory')
    if any(r.get('status')=='running' for r in manifest.get('runs',[])):
        parser.error('Campaign has a running trial; inspect it before starting another')
    model_root=Path(os.environ.get('MUSE_ROOT',
        '/metadisk/may/muse-glimmer' if Path('/metadisk/may/muse-glimmer').exists()
        else str(Path.home()/'.local/share/muse-glimmer')))
    model_lock=(model_root/'.baseline_queue.lock').open('a')
    try:
        fcntl.flock(model_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another baseline or pointing experiment is using the model')
    server={}
    if not remote:
        with httpx.Client(trust_env=False,timeout=30) as client:
            client.get(base_url+'/models').raise_for_status()
            response=client.get(base_url.removesuffix('/v1')+'/props')
            response.raise_for_status()
            server=response.json()
    directory=('deepseek_task00_' if remote else 'task00_')+args.effort+'_'+time.strftime('%Y%m%d-%H%M%S')
    folder=root/directory; folder.mkdir()
    system=image_instructions(prompt(args.profile),variant['clean_images']).replace('80 model decisions and 900 simulator steps',
                           f'{args.max_calls} model decisions and {args.max_steps} simulator steps')
    if remote: system=system.replace('Muse Glimmer','DeepSeek V4.1-Flash')
    budget=BatchBudget(args.api_budget,folder/'api_budget.json') if remote else None
    protocol=dict(model=model,provider=args.provider,
        api_budget_usd=args.api_budget if remote else None,
        requested_reasoning_effort=args.effort,
        reasoning_effort_verification='Requested max; provider internal effort is not independently observable' if remote else 'Local chat template',wire='chat',reasoning_effort=args.effort,task_id=0,
        task_name=LIBERO_10_TASKS[0][0],instruction=LIBERO_10_TASKS[0][1],initial_state_ids=args.inits,
        harness='open-gemini single-Panda automatic-check adaptation',
        baseline_protocol=False,zero_shot=True,demonstrations=False,prior_learnings=None,
        model_inputs='RGB overlays, measured robot state, table plane; optional rendered-depth deprojection',
        image_resolution=[512,512],image_horizon=2,max_steps=args.max_steps,
        max_api_attempts_per_episode=args.max_calls,max_output_tokens=args.max_output_tokens,
        system_prompt=system,checks='Panda damped-Jacobian IK + joint-limit and sampled MuJoCo contacts',
        checker_uses_privileged_scene_geometry=True,object_poses_exposed_to_policy=False,
        human_review=False,bimanual=False,checker_is_upstream_gap_guard=False,
        upstream_commit='470e1d2f943e3f8bb832b9aef28e4310e81344bc',
        memory='Full episode text history; latest two image pairs; fresh start each episode',
        checker_scope='Per-segment Cartesian IK and sampled joint contacts; no full-program preview or gripper sweep',
        checker_version='bilateral-carry-preview-v3',
        holding_detection='Measured finger-gap proxy, no learned holding detector',
        execution_profile='wrist-targeting-v3' if args.profile=='wrist-v3' else Session.execution_profile,
        targeting_profile=args.profile,trial_label=args.trial_label,
        experiment_variant=args.variant,experiment_switches=variant,
        experiment_source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/llm_agent_memory/harness/session.py',
             Path(__file__).resolve().parents[1]/'src/llm_agent_memory/harness/experiments.py',
             Path(__file__).resolve().parents[1]/'src/llm_agent_memory/harness/haimaker.py']},
        generation_seed=args.generation_seed,seed_schedule='base + decision number' if args.generation_seed is not None else None,
        compact_observations=args.profile=='wrist-v3',max_consecutive_look_only=2,
        server_context_tokens=server.get('default_generation_settings',{}).get('n_ctx'),
        server_generation_defaults=server.get('default_generation_settings',{}),
        server_build=server.get('build_info'),
        motion_settings=dict(descent_speed_m_s=.04,retract_speed_cap_m_s=.08,
            fast_descent=args.profile=='wrist-v3',final_slow_descent_m=.02,
            default_grasp_approach_height_m=.05 if args.profile=='wrist-v3' else .08,
            gripper_max_steps=20,settle_max_steps=5,
            measured_position_tolerance_m=.004,measured_orientation_tolerance_deg=2))
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run=dict(directory=directory,model=model,effort=args.effort,task_id=0,
             initial_state_ids=args.inits,status='running',trial_label=args.trial_label)
    manifest['runs'].append(run)
    results=[]
    def save():
        for path,data in [(folder/'summary.json',dict(protocol=protocol,results=results)),(manifest_path,manifest)]:
            temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)); temp.replace(path)
    save()
    try:
        for ep,state in enumerate(args.inits):
            episode_folder=folder/f'episode_{ep:02d}'; episode_folder.mkdir()
            session=Session(episode_folder,args.max_steps,
                compact_observations=args.profile=='wrist-v3',
                efficient_contact=args.profile=='wrist-v3',clean_images=variant['clean_images'],
                image_tone=variant.get('image_tone','identity'))
            transport=(HaimakerCapture(folder/'api',key,budget,max_tokens=args.max_output_tokens) if remote
                else Capture(folder/'api',args.effort,max_tokens=args.max_output_tokens))
            transport.episode=ep
            messages=[dict(role='system',content=system)]
            error=None
            monitor=RecoveryMonitor() if variant['recovery'] else None
            try:
                session.reset(libero_10_task(0,num_scenes=1,start=state).scenes[0])
                with httpx.Client(transport=transport,trust_env=False,timeout=1800) as client:
                    for call in range(1,args.max_calls+1):
                        start_step=session.steps
                        image_context=(session.crop,session.steps)
                        obs,images=session.observe(call)
                        if monitor and monitor.advice(): obs['recovery']=monitor.advice()
                        content=[dict(type='text',text='Task: '+LIBERO_10_TASKS[0][1]+'\nObservation:\n'+json.dumps(obs))]
                        for name,image in zip(['agentview','wrist'],images):
                            content.extend([dict(type='text',text=name+' camera'),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image).decode()))])
                        messages.append(dict(role='user',content=content))
                        # Full text history; images only from two most recent observations.
                        users=[i for i,m in enumerate(messages) if m['role']=='user']
                        for i in users[:-2]:
                            if isinstance(messages[i]['content'],list):
                                messages[i]['content']=[p for p in messages[i]['content'] if p['type']!='image_url']
                        body=dict(model=model,messages=messages)
                        if args.generation_seed is not None: body['seed']=args.generation_seed+call
                        response=client.post(base_url+'/chat/completions',json=body)
                        response.raise_for_status(); payload=response.json(); choice=payload['choices'][0]
                        message=choice['message']; text=message.get('content') or ''
                        messages.append(dict(role='assistant',content=text))
                        code=None
                        if choice.get('finish_reason')=='length':
                            session.feedback='Response truncated; nothing executed. Return a shorter complete command.'
                        else:
                            try:
                                endpoint,code=extract_curl(text)
                                if endpoint=='obs': session.request_observation()
                                else:
                                    rejection=monitor.before(code,session.last['eef_state'][:9],image_context) if monitor else None
                                    if rejection: session.feedback=rejection
                                    else: session.execute(code)
                            except (ValueError,ArithmeticError) as exc:
                                session.feedback='plan_error: '+str(exc)
                        if monitor:
                            monitor.after(code,session.feedback,start_step,session.steps,session.last['eef_state'][:9],image_context)
                        with (episode_folder/'turns.jsonl').open('a') as log:
                            log.write(json.dumps(dict(call=call,start_step=start_step,step=session.steps,
                                response=text,feedback=session.feedback))+'\n')
                        if session.done: break
                status='success' if session.success else 'failure'
            except ProviderStop as exc:
                error=str(exc);status='incomplete_budget' if 'budget' in error else 'error'
            except Exception as exc:
                error=repr(exc); status='error'
            finally:
                session.close()
            results.append(dict(episode=ep,init_state=state,status=status,environment_success=session.success,
                steps=session.steps,api_attempts=transport.calls,stop_detail=error,
                video=str(session.robot.episode_dir/'rollout.mp4') if session.robot.episode_dir else None))
            save(); print(json.dumps(results[-1]),flush=True)
            if error:
                raise RuntimeError('Infrastructure error: '+error)
        run['status']='completed'
    except BaseException:
        run['status']='interrupted'; raise
    finally:
        save()


if __name__=='__main__': main()

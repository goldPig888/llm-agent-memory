"""Open Gemini harness adapted to one Panda in LIBERO; separate from paper baseline."""
import argparse
import base64
import fcntl
import json
import time
from pathlib import Path

import httpx
from inspect_robots_libero.tasks import libero_10_task, LIBERO_10_TASKS
from llm_agent_memory.harness.session import Session
from llm_agent_memory.harness.upstream import SYSTEM_PROMPT, extract_curl
from run_local import Capture, local_url


def prompt():
    text=SYSTEM_PROMPT.replace('Gemini 4 Argon controlling a real 6-DoF YAM robot arm',
                               'Muse Glimmer controlling a simulated 7-DoF Panda robot arm')
    text=text.replace('A human operator previews every motion segment in 3D and must approve it\nbefore the arm moves. If they reject, their feedback is in the response --\nfollow it.',
        'An automatic Panda IK and swept-contact checker previews each segment.\nRejected segments return IK_FAILURE or COLLISION; correct the proposal.')
    text=text.replace('Arm base frame: +x forward (away from the arm base), +y left, +z up.',
                      'LIBERO world frame: x and y follow the calibrated grid, +z up.')
    text=text.replace('Overhead camera (calibrated)', 'Fixed agentview camera (calibrated; oblique rather than overhead)')
    text=text.replace('from stereo depth (noisy, ~1-2 cm)', 'from the simulated camera depth sensor')
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
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/muse_harness_xhigh'))
    parser.add_argument('--append',action='store_true',help='Preserve earlier trials and append a new run')
    parser.add_argument('--trial-label',default='')
    parser.add_argument('--inits',type=int,nargs='+',default=list(range(10)),choices=range(50))
    parser.add_argument('--effort',choices=['low','medium','high','xhigh'],default='xhigh')
    parser.add_argument('--max-calls',type=int,default=80)
    parser.add_argument('--max-steps',type=int,default=900)
    parser.add_argument('--max-output-tokens',type=int,default=16384)
    parser.add_argument('--base-url',type=local_url,default='http://127.0.0.1:8081/v1')
    args=parser.parse_args()
    if min(args.max_calls,args.max_steps,args.max_output_tokens)<1: parser.error('limits must be positive')
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
    with httpx.Client(trust_env=False,timeout=30) as client:
        client.get(args.base_url+'/models').raise_for_status()
        response=client.get(args.base_url.removesuffix('/v1')+'/props')
        response.raise_for_status()
        server=response.json()
    directory='task00_'+args.effort+'_'+time.strftime('%Y%m%d-%H%M%S')
    folder=root/directory; folder.mkdir()
    system=prompt().replace('80 model decisions and 900 simulator steps',
                           f'{args.max_calls} model decisions and {args.max_steps} simulator steps')
    protocol=dict(model='muse-glimmer',wire='chat',reasoning_effort=args.effort,task_id=0,
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
        holding_detection='Measured finger-gap proxy, no learned holding detector',
        execution_profile=Session.execution_profile,trial_label=args.trial_label,
        server_context_tokens=server.get('default_generation_settings',{}).get('n_ctx'),
        server_generation_defaults=server.get('default_generation_settings',{}),
        server_build=server.get('build_info'),
        motion_settings=dict(descent_speed_m_s=.04,retract_speed_cap_m_s=.08,
            gripper_max_steps=20,settle_max_steps=5,
            measured_position_tolerance_m=.004,measured_orientation_tolerance_deg=2))
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    run=dict(directory=directory,model='muse-glimmer',effort=args.effort,task_id=0,
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
            session=Session(episode_folder,args.max_steps)
            transport=Capture(folder/'api',args.effort,max_tokens=args.max_output_tokens); transport.episode=ep
            messages=[dict(role='system',content=system)]
            error=None
            try:
                session.reset(libero_10_task(0,num_scenes=1,start=state).scenes[0])
                with httpx.Client(transport=transport,trust_env=False,timeout=1800) as client:
                    for call in range(1,args.max_calls+1):
                        start_step=session.steps
                        obs,images=session.observe(call)
                        content=[dict(type='text',text='Task: '+LIBERO_10_TASKS[0][1]+'\nObservation:\n'+json.dumps(obs))]
                        for name,image in zip(['agentview','wrist'],images):
                            content.extend([dict(type='text',text=name+' camera'),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image).decode()))])
                        messages.append(dict(role='user',content=content))
                        # Full text history; images only from two most recent observations.
                        users=[i for i,m in enumerate(messages) if m['role']=='user']
                        for i in users[:-2]:
                            if isinstance(messages[i]['content'],list):
                                messages[i]['content']=[p for p in messages[i]['content'] if p['type']!='image_url']
                        body=dict(model='muse-glimmer',messages=messages)
                        response=client.post(args.base_url+'/chat/completions',json=body)
                        response.raise_for_status(); payload=response.json(); choice=payload['choices'][0]
                        message=choice['message']; text=message.get('content') or ''
                        messages.append(dict(role='assistant',content=text))
                        if choice.get('finish_reason')=='length':
                            session.feedback='Response truncated; nothing executed. Return a shorter complete command.'
                        else:
                            try:
                                endpoint,code=extract_curl(text)
                                if endpoint=='obs': session.feedback='New observation requested; no motion.'
                                else: session.execute(code)
                            except (ValueError,ArithmeticError) as exc:
                                session.feedback='plan_error: '+str(exc)
                        with (episode_folder/'turns.jsonl').open('a') as log:
                            log.write(json.dumps(dict(call=call,start_step=start_step,step=session.steps,
                                response=text,feedback=session.feedback))+'\n')
                        if session.done: break
                status='success' if session.success else 'failure'
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

import copy
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from llm_agent_memory.harness.experiments import RecoveryMonitor, VARIANTS, image_instructions
from llm_agent_memory.harness.session import Session, panda_orientation


def test_factorial_variants_change_only_declared_general_mechanisms():
    assert {'control','recovery','clean-images','combined'}.issubset(VARIANTS)
    assert VARIANTS['control']==dict(recovery=False,clean_images=False)
    assert VARIANTS['combined']==dict(recovery=True,clean_images=True)
    assert image_instructions('original',False)=='original'
    assert 'NO drawn grid' in image_instructions('original',True)


def test_visual_variants_change_only_tone_not_prompt_recovery_or_geometry():
    from llm_agent_memory.harness.experiments import tone_image
    for variant in ['clean-bright','clean-contrast']:
        switches=dict(VARIANTS[variant]);switches.pop('image_tone')
        assert switches==VARIANTS['clean-images']
    image=Image.fromarray(np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(10,1,3)))
    assert tone_image(image) is image
    bright=np.asarray(tone_image(image,'gamma-0.8'))
    assert bright.shape==np.asarray(image).shape
    assert np.all(np.diff(bright[0,:,0].astype(int))>=0)
    assert bright[0,64,0]>64 and bright[0,0,0]==0 and bright[0,255,0]==255
    np.testing.assert_array_equal(bright[:,:,0],bright[:,:,1])
    flat=Image.new('RGB',(25,25),(30,30,30))
    assert tone_image(flat,'autocontrast-1pct').size==flat.size
    with pytest.raises(ValueError):tone_image(image,'unknown')


def test_rejected_program_guard_ignores_formatting_and_resets_after_robot_motion():
    m=RecoveryMonitor();pose=np.zeros(9)
    code='robot.move_by(dz=0.1)'
    assert m.before(code,pose) is None
    m.after(code,json.dumps(dict(status='IK_FAILURE')),10,10,pose)
    blocked=m.before('robot.move_by( dz = 0.1 ) # retry',pose)
    assert json.loads(blocked)['status']=='repeat_rejected'
    assert m.before('robot.move_by(dz=0.02)',pose) is None
    pose[2]=.02
    assert m.before(code,pose) is None


def test_zoom_does_not_erase_rejected_motion_history():
    m=RecoveryMonitor();pose=np.zeros(9);code='robot.goto(x=0,y=0,z=.6)'
    m.after(code,'{"status":"COLLISION"}',0,0,pose)
    m.after('robot.look_at(x=0,y=0,z=.4)','{"status":"completed"}',0,0,pose)
    assert m.before(code,pose) is not None
    assert m.advice()['decisions_without_motion']==2
    assert m.advice()['rejected_programs_at_current_pose']==1


def test_successful_or_invalid_plan_is_not_remembered_as_motion_rejection():
    m=RecoveryMonitor();pose=np.zeros(9)
    m.after('robot.home()','{"status":"completed"}',0,10,pose)
    assert m.before('robot.home()',pose) is None
    m.after('invalid(','plan_error: invalid',10,10,pose)
    with pytest.raises(ValueError,match='syntax'):m.before('invalid(',pose)


def test_pixel_program_is_reconsidered_after_camera_crop_changes():
    m=RecoveryMonitor();pose=np.zeros(9)
    code='x,y,z=robot.deproject_depth(.5,.5)\nrobot.goto(x=x,y=y,z=z+.1)'
    old=((0,0,512,512),0);new=((100,100,356,356),0)
    m.after(code,'{"status":"IK_FAILURE"}',0,0,pose,old)
    assert m.before(code,pose,old) is not None
    assert m.before(code,pose,new) is None


def test_clean_image_preserves_exact_crop_and_camera_geometry(tmp_path):
    s=object.__new__(Session);r=panda_orientation(0)
    cam=dict(K=[[300,0,256],[0,300,256],[0,0,1]],R=np.diag([1.,-1.,-1.]),position=[0,0,1.])
    pixels=np.zeros((512,512,3),np.uint8);pixels[:,:,0]=np.arange(512,dtype=np.uint16)[None,:] % 256
    s.last=dict(eef_state=np.r_[0,0,.6,r[:,:2].T.reshape(-1),0].tolist(),
        gripper_qpos=[.04,-.04],joint_positions=[0]*7,table_z=.4,
        cameras={'agentview':cam,'robot0_eye_in_hand':copy.deepcopy(cam)},
        images={'agentview_image':pixels,'robot0_eye_in_hand_image':pixels.copy()})
    s.folder=tmp_path;s.crop=(100,100,356,356);s.grid_z=.4;s.last_target=None
    s.holding=False;s.feedback='Ready';s.steps=0;s.max_steps=900;s.execution_profile='test'
    s.look_only_count=0;s.grasp_offset=None;s.compact_observations=True;s.clean_images=True
    before=copy.deepcopy(s.last['cameras']);obs,images=s.observe(1)
    expected=Image.fromarray(pixels).crop(s.crop).resize((512,512))
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(images[0]))),np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(images[1]))),pixels)
    np.testing.assert_array_equal(s.last['cameras']['agentview']['R'],before['agentview']['R'])
    assert obs['crop_pixels']==s.crop


def test_trial_metrics_distinguish_proposal_execution_and_environment_success(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/queue_harness_ablations.py'
    spec=importlib.util.spec_from_file_location('queue_harness_ablations',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    episode=tmp_path/'episode_00';episode.mkdir()
    rows=[dict(call=1,start_step=0,step=0,response='robot.grasp(x=0,y=0,z=.4)',feedback='{"status":"IK_FAILURE","executed":[]}'),
          dict(call=2,start_step=0,step=20,response='robot.grasp(x=0,y=0,z=.4)',feedback='{"status":"completed","executed":[{"primitive":"grasp"}]}')]
    (episode/'turns.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    result=module.metrics(tmp_path)
    assert result['first_grasp_proposed_call']==1
    assert result['first_grasp_primitive_completed_call']==2
    assert result['environment_success'] is False


def test_ablation_viewer_includes_queued_variants_and_escapes_text(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/ablation_report.py'
    spec=importlib.util.spec_from_file_location('ablation_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=dict(protocol=dict(variants=['control','recovery'],switches={'control':{},'recovery':{}}),
        trials=[dict(variant='control',status='completed',metrics={'results':[{}],'environment_success':False})])
    (tmp_path/'general-ablation-test.json').write_text(json.dumps(data))
    output=module.collect_ablations(tmp_path)
    assert output['campaigns'][0]['trials'][1]['status']=='queued'
    rendered=module.render_ablations(output).decode()
    assert 'pending' in rendered and 'recovery' in rendered
    output['campaigns'][0]['name']='<script>'
    assert '<script>' not in module.render_ablations(output).decode()

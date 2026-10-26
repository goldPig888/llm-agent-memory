import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from llm_agent_memory.harness.plan_program import parse_plan,PlanError
from llm_agent_memory.harness.session import Session
from llm_agent_memory.harness.upstream import SIGNATURES


def fake_session():
    s=object.__new__(Session)
    camera=dict(K=[[300,0,256],[0,300,256],[0,0,1]],R=np.diag([1.,-1.,-1.]),position=[0,0,1.])
    second=copy.deepcopy(camera);second['position']=[.2,0,1.]
    s.last={'cameras':{'agentview':camera,'robot0_eye_in_hand':second},
        'depths':{'agentview':np.full((512,512),.6),'robot0_eye_in_hand':np.full((512,512),.6)}}
    s.crop=(100,100,356,356)
    return s


def test_wrist_deprojection_uses_wrist_pose_and_full_image():
    s=fake_session()
    x,y=s.deproject(256/511,256/511,.4,camera='wrist')
    np.testing.assert_allclose([x,y],[.2,0],atol=1e-10)
    x,y,z=s.deproject_depth(256/511,256/511,camera='wrist')
    np.testing.assert_allclose([x,y,z],[.2,0,.4],atol=1e-10)


def test_wrist_target_program_accepts_only_literal_camera_names():
    captured=[]
    def deproject(**kwargs):captured.append(kwargs);return (.1,.2)
    calls=parse_plan('x,y=robot.deproject(.5,.5,.45,camera="wrist")\nrobot.grasp(x=x,y=y,z=.45)',
        SIGNATURES,{'deproject':deproject})
    assert captured[0]['camera']=='wrist' and calls[0].kwargs['x']==.1
    for bad in ['"other"','__import__("os")','"wrist"+"x"']:
        with pytest.raises(PlanError):parse_plan(f'x,y=robot.deproject(.5,.5,.45,camera={bad})\nrobot.home()',SIGNATURES,{'deproject':deproject})


def test_third_zoom_only_plan_is_not_executed():
    s=fake_session();s.signatures=copy.deepcopy(SIGNATURES);s.look_only_count=2
    s.max_look_only=2;s.steps=0;s.max_steps=900
    feedback=s.execute('robot.look_at(x=0,y=0,z=.4)')
    assert 'observation_limit' in feedback and s.steps==0


def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_pointing_scoring_and_input_separation(tmp_path):
    from PIL import Image
    pointing=module('run_pointing')
    image=tmp_path/'image.png';Image.new('RGB',(10,10),'red').save(image)
    mask=np.zeros((10,10),bool);mask[4:6,4:6]=True;np.save(tmp_path/'mask.npy',mask)
    sample={'image':'image.png','instruction':'Find alphabet soup','scoring_secret':'DO_NOT_SEND'}
    message=pointing.input_message(sample,tmp_path)
    assert 'DO_NOT_SEND' not in str(message)
    score=pointing.score_point([.5,.5],{'visible_pixels':4,'mask':'mask.npy','projected_center_px':[4.5,4.5]},tmp_path)
    assert score['hit_visible_object'] and score['pixel_error']==0
    assert pointing.parse_point('{"visible":false}') is None
    with pytest.raises(ValueError):pointing.parse_point('{"u":2,"v":0.1}')


def test_height_error_sensitivity_depends_on_camera_angle():
    collect=module('collect_pointing')
    for degrees,expected in [(32,1.6003),(87,.0524)]:
        a=np.deg2rad(degrees);rotation=np.eye(3);rotation[:,2]=[np.cos(a),0,-np.sin(a)]
        assert collect.camera_geometry({'R':rotation})['horizontal_error_per_height_error']==pytest.approx(expected,abs=.001)


def test_relative_place_uses_measured_grasp_offset():
    from llm_agent_memory.harness.session import panda_orientation
    s=fake_session();r=panda_orientation(0)
    s.last.update(eef_state=np.r_[0,0,.7,r[:,:2].T.reshape(-1),1].tolist(),table_z=.4)
    s.signatures=copy.deepcopy(SIGNATURES);s.look_only_count=0;s.max_look_only=2
    s.steps=0;s.max_steps=900;s.done=False;s.success=False;s.holding=True
    s.grasp_offset=.04;s.grasp_approach=np.array([0,0,-1]);s.last_check=None;s.efficient_contact=False
    moves=[];s._move=lambda xyz,*args,**kwargs:moves.append(xyz)
    s._gripper=lambda closed:None
    feedback=s.execute('robot.place_relative(x=.1,y=.2,support_z=.5)')
    assert 'completed' in feedback
    assert moves[1][2]==pytest.approx(.55)
    assert s.grasp_offset is None


def test_carried_pose_transform_preserves_grip_relative_offset():
    from llm_agent_memory.harness.worker_support import attachment_pose
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('z',90,degrees=True).as_matrix()
    position,rotation=attachment_pose(np.array([1.,2.,3.]),r,
        {'offset':np.array([.1,0,0]),'rotation':np.eye(3)})
    np.testing.assert_allclose(position,[1,2.1,3]);np.testing.assert_allclose(rotation,r)


def test_fast_descent_keeps_last_two_centimeters_slow():
    from llm_agent_memory.harness.session import panda_orientation
    s=fake_session();r=panda_orientation(0)
    s.last.update(eef_state=np.r_[0,0,.7,r[:,:2].T.reshape(-1),0].tolist(),table_z=.41)
    s.signatures=copy.deepcopy(SIGNATURES);s.signatures['grasp']['approach_height']=.05
    s.look_only_count=0;s.max_look_only=2;s.steps=0;s.max_steps=900;s.done=False
    s.success=False;s.holding=False;s.grasp_offset=None;s.grasp_approach=None
    s.last_check=None;s.efficient_contact=True;moves=[]
    def move(xyz,rotation=None,speed=.1,**kwargs):
        moves.append((list(xyz),speed));s.last['eef_state'][:3]=list(xyz)
    def grip(closed):s.holding=closed
    s._move=move;s._gripper=grip
    feedback=s.execute('robot.grasp(x=.1,y=.2,z=.47,yaw_deg=0,support_z=.41)')
    assert 'completed' in feedback
    assert moves[1][0][2]==pytest.approx(.49) and moves[1][1]==.08
    assert moves[2][0][2]==pytest.approx(.47) and moves[2][1]==.04
    assert s.grasp_offset==pytest.approx(.06)


def test_v3_prompt_removes_overhead_only_instruction(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=module('run_harness');text=runner.prompt('wrist-v3')
    assert 'deproject through the overhead image only' not in text
    assert 'camera="wrist"' in text
    assert 'Agentview is cropped; wrist is full-frame' in text

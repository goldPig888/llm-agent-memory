import numpy as np
import pytest
from llm_agent_memory.harness.geometry import project, deproject
from llm_agent_memory.harness.plan_program import parse_plan, PlanError
from llm_agent_memory.harness.upstream import SIGNATURES, extract_curl
from llm_agent_memory.harness.session import panda_orientation, Session


@pytest.mark.parametrize('crop',[(0,0,512,512),(100,100,356,356)])
def test_projection_and_plane_deprojection_agree_after_zoom(crop):
    camera=dict(K=[[300,0,256],[0,300,256],[0,0,1]],R=np.diag([1.,-1.,-1.]),position=[0,0,1.])
    point=[.02,-.03,.4]
    pixel=project(point,camera)
    u=(pixel[0]-crop[0])/(crop[2]-crop[0]-1)
    v=(pixel[1]-crop[1])/(crop[3]-crop[1]-1)
    np.testing.assert_allclose(deproject(u,v,.4,camera,crop),point[:2],atol=1e-10)


def test_panda_jaw_axis_and_downward_approach():
    for yaw in (0,45,90,180):
        r=panda_orientation(yaw)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10)
        np.testing.assert_allclose(r[:,2],[0,0,-1],atol=1e-10)
        np.testing.assert_allclose(r[:2,0],[np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw))],atol=1e-10)
        assert np.linalg.det(r)>0


def test_upstream_parser_supports_pixel_target_program():
    calls=parse_plan('x,y=robot.deproject(u=.5,v=.5,z_plane=.45)\nrobot.grasp(x=x,y=y,z=.45)',
                     SIGNATURES,{'deproject':lambda **k:(.1,.2)})
    assert calls[0].name=='grasp'
    assert calls[0].kwargs['x']==.1
    assert calls[0].kwargs['approach_height']==.08


@pytest.mark.parametrize('code',['import os','robot.grasp(x=__import__("os"),y=0,z=.5)',
                                 'robot.__class__()','left.home()'])
def test_parser_does_not_execute_model_python(code):
    with pytest.raises(PlanError): parse_plan(code,SIGNATURES,{})


def test_curl_is_parsed_without_executing_shell():
    endpoint,code=extract_curl("Scene.\n# Thought: observe.\ncurl -s -X POST http://127.0.0.1:8081/plan --data-binary @- <<'EOF'\nrobot.home()\nEOF")
    assert endpoint=='plan' and code.strip()=='robot.home()'


def test_out_of_workspace_motion_is_rejected_before_robot_access():
    session=object.__new__(Session)
    session.last={'table_z':.41}
    with pytest.raises(ValueError,match='COLLISION'): session._move([0,0,.3])
    with pytest.raises(ValueError,match='IK_FAILURE'): session._move([2,0,.6])

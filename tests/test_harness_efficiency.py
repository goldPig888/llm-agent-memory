from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from llm_agent_memory.harness.session import Session, panda_orientation


def session_at_pose():
    s=object.__new__(Session)
    pose=np.r_[0,0,.7,panda_orientation(0)[:,:2].T.reshape(-1),0]
    s.last={'eef_state':pose.tolist(),'gripper_qpos':[.04,-.04],'table_z':.41}
    s.steps=0; s.max_steps=900; s.done=False; s.holding=False
    s.robot=Mock(); s.last_target=None
    return s


def test_already_open_uses_no_physics_steps():
    s=session_at_pose(); s._step=Mock()
    s._gripper(False)
    s._step.assert_not_called(); s.robot._rpc.assert_not_called()


def test_closing_waits_for_measured_stability_and_command():
    s=session_at_pose(); s.robot._rpc.return_value={'verdict':'OK'}
    ticks=[]
    def step(pose,gripper):
        ticks.append(1)
        s.last['eef_state'][9]=min(1,len(ticks)/10)
        # Fingers stop against an object before command ramp completes.
        gap=max(.02,.04-.005*len(ticks))
        s.last['gripper_qpos']=[gap,-gap]
    s._step=step
    s._gripper(True)
    assert 10<=len(ticks)<=20


def test_already_reached_pose_uses_no_physics_steps():
    s=session_at_pose(); s._step=Mock()
    s._move([0,0,.7])
    s._step.assert_not_called(); s.robot._rpc.assert_not_called()


def test_step_limit_is_still_hard(tmp_path):
    s=session_at_pose(); s.folder=tmp_path; s.steps=899; s.motion_log=[]
    s.robot.last=s.last
    s.robot.step.return_value=SimpleNamespace(terminated=False,termination_reason=None)
    pose=np.asarray(s.last['eef_state'][:9])
    s._step(pose,0); s._step(pose,0)
    assert s.steps==900 and s.done
    assert s.robot.step.call_count==1

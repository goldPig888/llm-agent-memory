"""Model-free geometry and preview-restoration check; not a benchmark trial."""
import argparse
import json
from pathlib import Path

import numpy as np

from llm_agent_memory.harness.worker_support import calibrate,check_targets,held_attachment
from llm_agent_memory.harness.geometry import deproject,project


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(0)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
        camera_heights=512,camera_widths=512,camera_depths=True)
    report={}
    try:
        env.seed(142);env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(10):obs,_,_,_=env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
        data=calibrate(env,obs);sim=env.env.sim;robot=env.env.robots[0]
        camera=data['cameras']['robot0_eye_in_hand'];xy=deproject(.5,.5,data['table_z'],camera)
        pixel=project([*xy,data['table_z']],camera)
        report['wrist_round_trip_error_px']=float(np.linalg.norm(pixel-[255.5,255.5]))
        pose=np.r_[robot.controller.ee_pos,robot.controller.ee_ori_mat[:,:2].T.reshape(-1)]
        before=sim.get_state().flatten().copy();verdict=check_targets(env,[pose])
        report['identity_preview']=verdict
        report['preview_restores_state']=bool(np.array_equal(before,sim.get_state().flatten()))
        report['carry_without_contact']=check_targets(env,[pose],carry=True)
        # Synthetic attachment fixture: move one object between stationary jaws,
        # without a robot rollout or any model access to its identity/pose.
        name=next(name for name in env.env.objects_dict if 'alphabet_soup' in name)
        obj=env.env.objects_dict[name];start=sim.model.get_joint_qpos_addr(obj.joints[0])[0]
        sid=sim.model.site_name2id(robot.gripper.important_sites['grip_site'])
        fingers=[sim.model.get_joint_qpos_addr(joint) for joint in robot.gripper.joints]
        fixture=sim.get_state();attachment=None
        for gap in np.linspace(.04,.005,40):
            sim.set_state(fixture)
            sim.data.qpos[start:start+3]=sim.data.site_xpos[sid]
            sim.data.qpos[fingers[0]]=gap;sim.data.qpos[fingers[1]]=-gap;sim.forward()
            attachment=held_attachment(env,robot,sid)
            if attachment is not None:break
        report['bilateral_contact_fixture_found']=attachment is not None
        if attachment is not None:
            before=sim.get_state().flatten().copy()
            report['attached_preview']=check_targets(env,[pose],fingers_may_touch=True,carry=True)
            report['attached_preview_restores_state']=bool(np.array_equal(before,sim.get_state().flatten()))
            # Put a different free object in the carrying path, exclusively in
            # this synthetic checker fixture, to exercise object/world contact.
            for other_name,other in env.env.objects_dict.items():
                if other_name==name or len(other.joints)!=1:continue
                address=sim.model.get_joint_qpos_addr(other.joints[0])
                if not isinstance(address,tuple) or address[1]-address[0]!=7:continue
                target=pose.copy();target[0]+=.12
                sim.data.qpos[address[0]:address[0]+3]=sim.data.body_xpos[env.env.obj_body_id[name]]+[.12,0,0]
                sim.forward();before=sim.get_state().flatten().copy()
                report['carried_obstacle_preview']=check_targets(env,[target],fingers_may_touch=True,carry=True)
                report['obstacle_preview_restores_state']=bool(np.array_equal(before,sim.get_state().flatten()))
                break
        report['passed']=bool(report['wrist_round_trip_error_px']<1e-6 and report['preview_restores_state']
            and verdict['verdict']=='OK' and report['carry_without_contact']['verdict']=='CARRY_UNVERIFIED'
            and attachment is not None and report.get('attached_preview_restores_state')
            and report.get('carried_obstacle_preview',{}).get('verdict')=='COLLISION'
            and any('carried_object' in pair for pair in report['carried_obstacle_preview'].get('contacts',[]))
            and report.get('obstacle_preview_restores_state'))
    finally:env.close()
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()

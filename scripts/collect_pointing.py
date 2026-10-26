"""Render initial states for pointing tests; keep scoring truth separate from inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from llm_agent_memory.harness.geometry import project
from llm_agent_memory.harness.worker_support import calibrate


def descendant_geoms(model,body):
    result=[]
    for gid in range(model.ngeom):
        bid=int(model.geom_bodyid[gid])
        while bid and bid!=body: bid=int(model.body_parentid[bid])
        if bid==body: result.append(gid)
    return result


def camera_geometry(camera):
    axis=np.asarray(camera['R'])[:,2]
    vertical=abs(float(axis[2]));horizontal=float(np.linalg.norm(axis[:2]))
    return dict(angle_below_horizontal_deg=float(np.rad2deg(np.arctan2(vertical,horizontal))),
                horizontal_error_per_height_error=horizontal/max(vertical,1e-9))


def render_segmentation(sim,camera):
    # robosuite 1.4's segmentation decoder overflows uint8 with NumPy 2.
    # Decode the rendered ID colors locally without patching the simulator.
    from robosuite.utils.binding_utils import _MjSim_render_lock
    with _MjSim_render_lock:
        context=sim._render_context_offscreen
        context.render(512,512,camera_id=sim.model.camera_name2id(camera),segmentation=True)
        rgb=context.read_pixels(512,512).astype(np.uint32)
        ids=rgb[:,:,0]+rgb[:,:,1]*256+rgb[:,:,2]*65536
        ids[ids>=context.scn.ngeom+1]=0
        lookup=np.full((context.scn.ngeom+1,2),-1,dtype=np.int32)
        for geom in context.scn.geoms[:context.scn.ngeom]:
            if geom.segid!=-1:lookup[geom.segid+1]=[geom.objtype,geom.objid]
        return lookup[ids][::-1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--states',type=int,nargs='+',default=list(range(50)),choices=range(50))
    p.add_argument('--cameras',nargs='+',choices=['agentview','wrist'],default=['agentview','wrist'])
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'images').mkdir();(root/'scoring').mkdir()
    # Import simulator only in the renderer; inference code does not import it.
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(0)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/
        task.problem_folder/task.bddl_file),camera_heights=512,camera_widths=512,camera_depths=True)
    samples=[];labels={};geometry={}
    try:
        for state in args.states:
            env.seed(142+state);env.reset()
            obs=env.set_init_state(suite.get_task_init_states(0)[state])
            for _ in range(10):obs,_,_,_=env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
            data=calibrate(env,obs);sim=env.env.sim
            names=[name for name in env.env.obj_body_id if 'alphabet_soup' in name]
            if len(names)!=1:raise ValueError(f'Expected one alphabet soup object, got {names}')
            bid=env.env.obj_body_id[names[0]];geoms=descendant_geoms(sim.model,bid)
            for camera in args.cameras:
                name='agentview' if camera=='agentview' else 'robot0_eye_in_hand'
                ident=f'state{state:02d}_{camera}'
                Image.fromarray(np.asarray(data['images'][name+'_image'],dtype=np.uint8)).save(root/'images'/f'{ident}.png')
                seg=render_segmentation(sim,name)
                mask=(seg[:,:,0]==5)&np.isin(seg[:,:,1],geoms)
                np.save(root/'scoring'/f'{ident}_mask.npy',mask)
                center=project(sim.data.body_xpos[bid],data['cameras'][name])
                labels[ident]=dict(projected_center_px=center.tolist() if center is not None else None,
                    visible_pixels=int(mask.sum()),mask=f'scoring/{ident}_mask.npy')
                samples.append(dict(id=ident,state=state,camera=camera,
                    image=f'images/{ident}.png',instruction='Point to the center of the alphabet soup can.'))
                geometry[ident]=camera_geometry(data['cameras'][name])
            print(f'Rendered state {state}; no scored physics steps.',flush=True)
    finally:env.close()
    (root/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    (root/'scoring/labels.json').write_text(json.dumps(labels,indent=2))
    (root/'camera_geometry.json').write_text(json.dumps(geometry,indent=2))
    (root/'protocol.json').write_text(json.dumps(dict(task=0,states=args.states,cameras=args.cameras,
        image_size=[512,512],neutral_settling_steps=10,scored_physics_steps=0,
        targets=['alphabet_soup'],ground_truth='Scoring only; never included in model requests'),indent=2))


if __name__=='__main__':main()

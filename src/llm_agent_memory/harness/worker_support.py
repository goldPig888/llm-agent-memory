"""Calibration and a Panda-specific IK/contact checker, inside the simulator.

Only camera calibration, robot proprioception, table plane, and rendered depth
leave the worker. Scene geometry is used by the checker, never as object poses
or grasp labels supplied to the policy. Preview restores the full simulator state.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def calibrate(env, obs):
    sim = env.env.sim
    robot = env.env.robots[0]
    images, cameras, depths = {}, {}, {}
    near = sim.model.vis.map.znear * sim.model.stat.extent
    far = sim.model.vis.map.zfar * sim.model.stat.extent
    for name in ('agentview', 'robot0_eye_in_hand'):
        rgb, depth = sim.render(camera_name=name, width=512, height=512, depth=True)
        # MuJoCo framebuffer origin is bottom-left; policy pixels are top-left.
        images[name + '_image'] = rgb[::-1].copy().tolist()
        depth = depth[::-1]
        depths[name] = (near / (1.0 - depth * (1.0 - near / far))).tolist()
        cid = sim.model.camera_name2id(name)
        rot = sim.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        focal = 256. / np.tan(np.deg2rad(sim.model.cam_fovy[cid]) / 2)
        cameras[name] = dict(K=[[focal, 0, 256], [0, focal, 256], [0, 0, 1]],
                             R=rot.tolist(), position=sim.data.cam_xpos[cid].tolist())
    table_z = None
    table_bounds=None
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ''
        if name == 'table_collision' or name.endswith('_table_collision'):
            table_z = float(sim.data.geom_xpos[gid, 2] + sim.model.geom_size[gid, 2])
            center=sim.data.geom_xpos[gid];size=sim.model.geom_size[gid]
            rotation=sim.data.geom_xmat[gid].reshape(3,3)
            corners=[center+rotation@(size*np.array([x,y,1])) for x in (-1,1) for y in (-1,1)]
            table_bounds=[point.tolist() for point in corners]
            break
    if table_z is None:
        arena = env.env.model.mujoco_arena
        table_z = float(np.asarray(arena.table_offset)[2])
    return dict(images=images, cameras=cameras, depths=depths, table_z=table_z,table_corners=table_bounds,
                joint_positions=sim.data.qpos[robot._ref_joint_pos_indexes].tolist())


def held_attachment(env,robot,sid):
    """Checker-only attachment from bilateral finger contact; never sent to policy."""
    sim=env.env.sim;touch={}
    sides={side:{sim.model.geom_name2id(name) for name in robot.gripper.important_geoms[side+'_finger']}
        for side in ('left','right')}
    for name,obj in env.env.objects_dict.items():
        if name not in env.env.obj_body_id or len(obj.joints)!=1:continue
        address=sim.model.get_joint_qpos_addr(obj.joints[0])
        if not isinstance(address,tuple) or address[1]-address[0]!=7:continue
        bid=env.env.obj_body_id[name];geoms=set()
        for gid in range(sim.model.ngeom):
            parent=int(sim.model.geom_bodyid[gid])
            while parent and parent!=bid:parent=int(sim.model.body_parentid[parent])
            if parent==bid:geoms.add(gid)
        contacts=set()
        for contact in sim.data.contact[:sim.data.ncon]:
            if contact.dist>.002:continue
            pair={int(contact.geom1),int(contact.geom2)}
            for side,fingers in sides.items():
                if pair&geoms and pair&fingers:contacts.add(side)
        if len(contacts)==2:touch[name]=(bid,address[0],geoms)
    if len(touch)!=1:return None
    bid,start,geoms=next(iter(touch.values()))
    grip_r=sim.data.site_xmat[sid].reshape(3,3)
    return dict(start=start,geoms=geoms,
        offset=grip_r.T@(sim.data.body_xpos[bid]-sim.data.site_xpos[sid]),
        rotation=grip_r.T@sim.data.body_xmat[bid].reshape(3,3))


def attachment_pose(position,rotation,attachment):
    return (position+rotation@attachment['offset'],rotation@attachment['rotation'])


def check_targets(env, targets, fingers_may_touch=False, carry=False):
    sim = env.env.sim
    robot = env.env.robots[0]
    pos_ids = np.asarray(robot._ref_joint_pos_indexes)
    vel_ids = np.asarray(robot._ref_joint_vel_indexes)
    joint_ids = [sim.model.joint_name2id(n) for n in robot.robot_model.joints]
    limits = sim.model.jnt_range[joint_ids]
    site = robot.gripper.important_sites['grip_site']
    sid = sim.model.site_name2id(site)
    saved = sim.get_state()
    attachment=held_attachment(env,robot,sid) if carry else None
    if carry and attachment is None:
        return {'verdict':'CARRY_UNVERIFIED','carried_object_preview':False}
    carried_geoms=attachment['geoms'] if attachment else set()
    free_bodies=set()
    for name,obj in env.env.objects_dict.items():
        if name not in env.env.obj_body_id or len(obj.joints)!=1:continue
        address=sim.model.get_joint_qpos_addr(obj.joints[0])
        if isinstance(address,tuple) and address[1]-address[0]==7:
            free_bodies.add(env.env.obj_body_id[name])
    free_geoms=set()
    for gid in range(sim.model.ngeom):
        parent=int(sim.model.geom_bodyid[gid])
        while parent and parent not in free_bodies:parent=int(sim.model.body_parentid[parent])
        if parent in free_bodies:free_geoms.add(gid)
    def contacts():
        pairs = {}
        for contact in sim.data.contact[:sim.data.ncon]:
            if contact.dist >= -0.002:
                continue
            names = [sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)]
            robot_mask = [n.startswith(('robot0_', 'gripper0_')) for n in names]
            if not any(robot_mask) and not ({int(contact.geom1),int(contact.geom2)}&carried_geoms):
                continue
            # Intended fingertip/object contact is allowed only for manipulation.
            if sum(robot_mask) == 1 and fingers_may_touch:
                robot_name = names[robot_mask.index(True)]
                other=int(contact.geom2) if robot_mask[0] else int(contact.geom1)
                if 'finger' in robot_name and other in free_geoms:
                    continue
            key=tuple(sorted(names))
            pairs[key]=min(pairs.get(key,0.),float(contact.dist))
        return pairs
    baseline = contacts()
    try:
        for waypoint,values in enumerate(targets):
            a = np.asarray(values, dtype=float)
            if a.shape != (9,) or not np.all(np.isfinite(a)):
                return {'verdict': 'IK_FAILURE','waypoint':waypoint+1}
            desired = a[3:].reshape(2, 3).T
            if np.linalg.norm(desired[:,0])<1e-6:
                return {'verdict':'IK_FAILURE','waypoint':waypoint+1}
            x = desired[:, 0] / np.linalg.norm(desired[:, 0])
            y = desired[:, 1] - x * (x @ desired[:, 1])
            if np.linalg.norm(y)<1e-6:
                return {'verdict':'IK_FAILURE','waypoint':waypoint+1}
            y /= np.linalg.norm(y)
            if not np.all(np.isfinite(np.r_[x,y])):
                return {'verdict':'IK_FAILURE','waypoint':waypoint+1}
            rotation = np.stack([x, y, np.cross(x, y)], axis=1)
            q_start = sim.data.qpos[pos_ids].copy()
            solved = False
            for _ in range(150):
                actual = sim.data.site_xmat[sid].reshape(3, 3)
                position_error = a[:3] - sim.data.site_xpos[sid]
                orientation_error = Rotation.from_matrix(rotation @ actual.T).as_rotvec()
                if np.linalg.norm(position_error) < .006 and np.linalg.norm(orientation_error) < .06:
                    solved = True
                    break
                jac = np.vstack([sim.data.get_site_jacp(site).reshape(3, -1)[:, vel_ids],
                                 sim.data.get_site_jacr(site).reshape(3, -1)[:, vel_ids]])
                delta = jac.T @ np.linalg.solve(jac @ jac.T + .002 * np.eye(6),
                                                np.r_[position_error, orientation_error])
                sim.data.qpos[pos_ids] = np.clip(sim.data.qpos[pos_ids] + np.clip(delta, -.12, .12),
                                                limits[:, 0] + .001, limits[:, 1] - .001)
                sim.forward()
            if not solved:
                return {'verdict': 'IK_FAILURE','waypoint':waypoint+1}
            end = sim.data.qpos[pos_ids].copy()
            n = max(2, int(np.max(np.abs(end - q_start)) / .04) + 1)
            for fraction in np.linspace(0, 1, n):
                sim.data.qpos[pos_ids] = q_start + fraction * (end - q_start)
                sim.forward()
                if attachment:
                    pos,rot=attachment_pose(sim.data.site_xpos[sid],sim.data.site_xmat[sid].reshape(3,3),attachment)
                    start=attachment['start'];sim.data.qpos[start:start+3]=pos
                    quat=Rotation.from_matrix(rot).as_quat()
                    sim.data.qpos[start+3:start+7]=quat[[3,0,1,2]];sim.forward()
                collision={pair for pair,depth in contacts().items()
                    if pair not in baseline or depth<baseline[pair]-.002}
                if collision:
                    def category(name):
                        if name.startswith(('robot0_','gripper0_')):
                            return 'robot_finger' if 'finger' in name else 'robot_link'
                        gid=sim.model.geom_name2id(name) if name else -1
                        return 'carried_object' if gid in carried_geoms else 'scene_geometry'
                    return {'verdict':'COLLISION','waypoint':waypoint+1,
                        'contacts':[[category(a),category(b)] for a,b in sorted(collision)],
                        'carried_object_preview':bool(attachment)}
        return {'verdict':'OK','carried_object_preview':bool(attachment)}
    finally:
        sim.set_state(saved)
        sim.forward()

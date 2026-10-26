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
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ''
        if name == 'table_collision' or name.endswith('_table_collision'):
            table_z = float(sim.data.geom_xpos[gid, 2] + sim.model.geom_size[gid, 2])
            break
    if table_z is None:
        arena = env.env.model.mujoco_arena
        table_z = float(np.asarray(arena.table_offset)[2])
    return dict(images=images, cameras=cameras, depths=depths, table_z=table_z,
                joint_positions=sim.data.qpos[robot._ref_joint_pos_indexes].tolist())


def check_targets(env, targets, fingers_may_touch=False):
    sim = env.env.sim
    robot = env.env.robots[0]
    pos_ids = np.asarray(robot._ref_joint_pos_indexes)
    vel_ids = np.asarray(robot._ref_joint_vel_indexes)
    joint_ids = [sim.model.joint_name2id(n) for n in robot.robot_model.joints]
    limits = sim.model.jnt_range[joint_ids]
    site = robot.gripper.important_sites['grip_site']
    sid = sim.model.site_name2id(site)
    saved = sim.get_state()
    def contacts():
        pairs = set()
        for contact in sim.data.contact[:sim.data.ncon]:
            if contact.dist >= -0.002:
                continue
            names = [sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)]
            robot_mask = [n.startswith(('robot0_', 'gripper0_')) for n in names]
            if not any(robot_mask):
                continue
            # Intended fingertip/object contact is allowed only for manipulation.
            if sum(robot_mask) == 1 and fingers_may_touch:
                robot_name = names[robot_mask.index(True)]
                if 'finger' in robot_name:
                    continue
            pairs.add(tuple(sorted(names)))
        return pairs
    baseline = contacts()
    try:
        for values in targets:
            a = np.asarray(values, dtype=float)
            if a.shape != (9,) or not np.all(np.isfinite(a)):
                return {'verdict': 'IK_FAILURE'}
            desired = a[3:].reshape(2, 3).T
            x = desired[:, 0] / np.linalg.norm(desired[:, 0])
            y = desired[:, 1] - x * (x @ desired[:, 1]); y /= np.linalg.norm(y)
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
                return {'verdict': 'IK_FAILURE'}
            end = sim.data.qpos[pos_ids].copy()
            n = max(2, int(np.max(np.abs(end - q_start)) / .04) + 1)
            for fraction in np.linspace(0, 1, n):
                sim.data.qpos[pos_ids] = q_start + fraction * (end - q_start)
                sim.forward()
                if contacts() - baseline:
                    return {'verdict': 'COLLISION'}
        return {'verdict': 'OK'}
    finally:
        sim.set_state(saved)
        sim.forward()

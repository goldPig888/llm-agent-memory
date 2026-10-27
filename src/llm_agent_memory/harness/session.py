"""Single-arm execution adapter for the upstream Code-as-Policies interface."""
import io
import json
import math
import copy
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from inspect_robots.types import Action
from .backend import HarnessEmbodiment
from .geometry import annotate, project, deproject
from .plan_program import parse_plan
from .upstream import SIGNATURES, orientation
from .experiments import tone_image


def panda_orientation(yaw, pitch=90., azimuth=0., roll=0.):
    yam = orientation(yaw, pitch, azimuth, roll)
    # Same world jaw/approach directions; different gripper-local axes.
    return np.stack([yam[:, 1], -yam[:, 0], yam[:, 2]], axis=1)


class Session:
    execution_profile = 'measured-settling-v2'
    def __init__(self, folder, max_steps=900, compact_observations=False,
                 max_look_only=2, efficient_contact=False, clean_images=False,
                 image_tone='identity'):
        self.folder = Path(folder)
        self.robot = HarnessEmbodiment(task_id=0, output_dir=self.folder)
        self.max_steps = max_steps
        self.steps = 0
        self.success = False
        self.done = False
        self.crop = (0, 0, 512, 512)
        self.grid_z = None
        self.last = None
        self.last_target = None
        self.holding = False
        self.feedback = 'Ready'
        self.motion_log = []
        self.clean_images=clean_images
        self.image_tone=image_tone
        self.compact_observations=compact_observations
        self.max_look_only=max_look_only
        self.look_only_count=0
        self.efficient_contact=efficient_contact
        self.grasp_offset=None
        self.last_check=None
        self.grasp_approach=None
        self.signatures=copy.deepcopy(SIGNATURES)
        if efficient_contact:
            self.execution_profile='wrist-targeting-v3'
            self.signatures['grasp']['approach_height']=.05

    def reset(self, scene):
        self.robot.reset(scene)
        self.last = self.robot.last
        self.grid_z = self.last['table_z']
        self.home = np.array(self.last['eef_state'][:9])
        if self.efficient_contact and self.last.get('table_corners'):
            pixels=[project(p,self.last['cameras']['agentview']) for p in self.last['table_corners']]
            if all(p is not None for p in pixels):
                pixels=np.asarray(pixels);low=np.floor(pixels.min(axis=0)-20).astype(int)
                high=np.ceil(pixels.max(axis=0)+20).astype(int)
                low=np.clip(low,0,480);high=np.clip(high,low+32,512)
                self.crop=(*low.tolist(),*high.tolist())

    def request_observation(self):
        self.look_only_count+=1
        self.feedback=('observation_limit: propose a pixel target and hover/motion next; '
            'additional observation-only turns do not improve geometry.' if self.look_only_count>self.max_look_only
            else 'New observation requested; no motion.')

    def observe(self, call):
        d = self.last
        position = np.array(d['eef_state'][:3])
        state = dict(eef_position=position.tolist(), orientation_rot6d=d['eef_state'][3:9],
                     commanded_pose=self.last_target, gripper_command=d['eef_state'][9],
                     finger_positions=d['gripper_qpos'], joint_positions=d['joint_positions'],
                     holding=self.holding, holding_estimate=self.holding, table_z=d['table_z'], grid_z=self.grid_z,
                     workspace=dict(x=[-.6,.6],y=[-.6,.6],z=[d['table_z'],1.6]),
                     steps=self.steps, max_steps=self.max_steps, feedback=self.feedback,
                     steps_remaining=max(0,self.max_steps-self.steps),
                     execution_profile=self.execution_profile,
                     crop_pixels=self.crop, world_frame='LIBERO world, meters, +z up')
        if self.last_target:
            state['tracking_error_m'] = float(np.linalg.norm(position - np.array(self.last_target[:3])))
        state['camera_targeting']={'agentview':{'crop_pixels':self.crop},
            'wrist':{'crop_pixels':[0,0,512,512]}}
        for alias,name in [('agentview','agentview'),('wrist','robot0_eye_in_hand')]:
            axis=np.asarray(d['cameras'][name]['R'])[:,2]
            state['camera_targeting'][alias]['angle_below_horizontal_deg']=round(
                float(np.rad2deg(np.arctan2(abs(axis[2]),np.linalg.norm(axis[:2])))),2)
        state['look_only_turns']=self.look_only_count
        state['grasp_height_offset_m']=self.grasp_offset
        if self.compact_observations:
            for field in ('joint_positions','workspace','world_frame','max_steps','holding_estimate','execution_profile'):
                state.pop(field,None)
        images = []
        for name, key in [('agentview','agentview_image'),('robot0_eye_in_hand','robot0_eye_in_hand_image')]:
            crop = self.crop if name == 'agentview' else (0,0,512,512)
            if self.clean_images:
                image=Image.fromarray(np.asarray(d['images'][key],dtype=np.uint8)).crop(crop).resize((512,512))
                image=tone_image(image,getattr(self,'image_tone','identity'))
            else:
                image = annotate(d['images'][key], d['cameras'][name], d['eef_state'], self.grid_z, crop,
                                 jaw_gap=float(np.abs(d['gripper_qpos']).sum()))
            out = self.folder / 'observations'; out.mkdir(exist_ok=True)
            image.save(out / f'call{call:02d}_{name}.png')
            buf=io.BytesIO(); image.save(buf,format='PNG'); images.append(buf.getvalue())
        return state, images

    def _camera(self,camera):
        if camera=='agentview': return 'agentview',self.crop
        if camera=='wrist': return 'robot0_eye_in_hand',(0,0,512,512)
        raise ValueError('camera must be agentview or wrist')

    def deproject(self, u, v, z_plane, camera='agentview'):
        name,crop=self._camera(camera)
        return deproject(u,v,z_plane,self.last['cameras'][name],crop)

    def deproject_depth(self, u, v, camera='agentview'):
        if not 0 <= u <= 1 or not 0 <= v <= 1:
            raise ValueError('pixels must be normalized 0..1')
        name,crop=self._camera(camera)
        px = crop[0]+u*(crop[2]-crop[0]-1)
        py = crop[1]+v*(crop[3]-crop[1]-1)
        cam=self.last['cameras'][name]
        depth=float(np.asarray(self.last['depths'][name])[int(round(py)),int(round(px))])
        if not np.isfinite(depth) or depth<=0:
            raise ValueError('invalid rendered depth at selected pixel')
        ray=np.linalg.solve(cam['K'],[px,py,1.])
        return tuple(np.asarray(cam['position'])+np.asarray(cam['R']) @ (depth*ray))

    def _update_holding(self):
        gap=float(np.abs(self.last['gripper_qpos']).sum())
        self.holding = bool(self.last['eef_state'][9] > .5 and .008 < gap < .075)

    def _step(self, pose, gripper):
        if self.done or self.steps>=self.max_steps:
            self.done=True
            return
        result=self.robot.step(Action(np.r_[pose,gripper]))
        self.last=self.robot.last; self.steps+=1
        self.success=result.terminated and result.termination_reason=='success'
        self.done=self.success or self.steps>=self.max_steps
        self.motion_log.append(dict(step=self.steps, target=np.r_[pose,gripper].tolist(),
                                    measured=self.last['eef_state'], success=self.success))
        with (self.folder/'motion.jsonl').open('a') as log:
            log.write(json.dumps(self.motion_log[-1])+'\n')
        self._update_holding()

    def _gripper(self, closed):
        pose=np.array(self.last['eef_state'][:9])
        gap=float(np.abs(self.last['gripper_qpos']).sum())
        command=float(self.last['eef_state'][9])
        # Opening an already fully open hand needs no physics rollout.
        if not closed and command < .02 and gap >= .075:
            self._update_holding()
            return
        # Gripper sweep has a different contact policy from free-space motion.
        check=self.robot._rpc({'op':'check','targets':[pose.tolist()], 'fingers_may_touch':True})
        if check['verdict']!='OK':
            raise ValueError(check['verdict'])
        previous=np.asarray(self.last['gripper_qpos'],float)
        stable=0
        for tick in range(20):
            self._step(pose, float(closed))
            if self.done: break
            current=np.asarray(self.last['gripper_qpos'],float)
            stable=stable+1 if np.max(np.abs(current-previous))<.0001 else 0
            previous=current.copy()
            command=float(self.last['eef_state'][9])
            command_reached=command>.98 if closed else command<.02
            if tick>=5 and command_reached and stable>=3:
                break

    def _rotation(self, yaw=math.nan,pitch=math.nan,azimuth=math.nan,roll=0.):
        current=np.array(self.last['eef_state'][3:9]).reshape(2,3).T
        z=np.cross(current[:,0],current[:,1])
        if all(np.isnan(v) for v in (yaw,pitch,azimuth)) and roll==0:
            return np.column_stack([current,z])
        if np.isnan(yaw): yaw=np.rad2deg(np.arctan2(current[1,0],current[0,0]))
        if np.isnan(pitch): pitch=np.rad2deg(np.arcsin(np.clip(-z[2],-1,1)))
        if np.isnan(azimuth): azimuth=np.rad2deg(np.arctan2(z[1],z[0]))
        return panda_orientation(yaw,pitch,azimuth,roll)

    def _move(self, xyz, rotation=None, speed=.10, fingers=False, guard=None):
        xyz=np.asarray(xyz,float)
        if not np.all(np.isfinite(xyz)) or not (-.6<=xyz[0]<=.6 and -.6<=xyz[1]<=.6 and self.last['table_z']-.002<=xyz[2]<=1.6):
            raise ValueError('COLLISION' if xyz[2]<self.last['table_z'] else 'IK_FAILURE')
        if not .005 <= speed <= .15:
            raise ValueError('speed must be 0.005..0.15 m/s')
        rotation=self._rotation() if rotation is None else rotation
        initial=np.array(self.last['eef_state'][:9])
        initial_r=initial[3:].reshape(2,3).T
        end=np.r_[xyz,rotation[:,:2].T.reshape(-1)]
        # Cartesian and angular waypoints are checked before executing a segment.
        distance=np.linalg.norm(xyz-initial[:3])
        angle=np.arccos(np.clip((np.trace(np.column_stack([initial_r,np.cross(initial_r[:,0],initial_r[:,1])]).T @ rotation)-1)/2,-1,1))
        # Do not execute or settle a segment already at its measured target.
        if distance <= .002 and angle <= np.deg2rad(1):
            self.last_target=end.tolist()
            return
        n=max(2,math.ceil(distance/.01),math.ceil(angle/np.deg2rad(5)))
        start_rotation=np.column_stack([initial_r,np.cross(initial_r[:,0],initial_r[:,1])])
        slerp=Slerp([0,1],Rotation.from_matrix([start_rotation,rotation]))
        def at(f):
            return np.r_[initial[:3]+(xyz-initial[:3])*f,slerp(f).as_matrix()[:,:2].T.reshape(-1)]
        waypoints=[at(f).tolist() for f in np.linspace(0,1,n+1)[1:]]
        check=self.robot._rpc({'op':'check','targets':waypoints,'fingers_may_touch':fingers or guard is not None,
                               'carry':self.holding})
        self.last_check=check
        if check['verdict']!='OK':
            raise ValueError(check['verdict'])
        grip=float(self.last['eef_state'][9])
        was_holding=self.holding
        self.last_target=end.tolist()
        count=max(1,math.ceil(distance/(speed/20)),math.ceil(angle/np.deg2rad(3)))
        for fraction in np.linspace(0,1,count+1)[1:]:
            target=at(fraction)
            self._step(target,grip)
            if self.done: return
            if was_holding and not self.holding:
                raise ValueError('dropped')
            if guard and np.linalg.norm(target[:3]-np.array(self.last['eef_state'][:3]))>guard:
                return  # Guarded contact uses measured tracking lag, not object poses.
        # Hold the endpoint briefly; OSC targets are not reached instantaneously.
        stable=0
        for _ in range(5):
            self._step(end,grip)
            if self.done: return
            if was_holding and not self.holding:
                raise ValueError('dropped')
            actual=np.asarray(self.last['eef_state'][:9])
            actual_r=actual[3:].reshape(2,3).T
            actual_rotation=np.column_stack([actual_r,np.cross(actual_r[:,0],actual_r[:,1])])
            rotation_error=np.arccos(np.clip((np.trace(actual_rotation.T@rotation)-1)/2,-1,1))
            reached=np.linalg.norm(actual[:3]-xyz)<=.004 and rotation_error<=np.deg2rad(2)
            stable=stable+1 if reached else 0
            if stable>=2:
                break
        error=np.linalg.norm(np.array(self.last['eef_state'][:3])-xyz)
        if error>.04:
            raise ValueError('blocked')

    def execute(self, code):
        calls=parse_plan(code,self.signatures,{'deproject':self.deproject,'deproject_depth':self.deproject_depth})
        look_only=all(call.name=='look_at' for call in calls)
        if look_only and self.look_only_count>=self.max_look_only:
            self.feedback=json.dumps(dict(status='observation_limit',
                message='Choose a pixel target and propose a hover/motion; another zoom-only turn was not executed.',
                steps_remaining=max(0,self.max_steps-self.steps)))
            return self.feedback
        self.look_only_count=self.look_only_count+1 if look_only else 0
        executed=[]
        try:
            for call in calls:
                if self.done: break
                start_step=self.steps
                n,k=call.name,call.kwargs
                if n=='look_at':
                    if not np.isnan(k['grid_z']): self.grid_z=k['grid_z']
                    if not 1<=k['zoom']<=8: raise ValueError('zoom must be 1..8')
                    pixel=project([k['x'],k['y'],k['z']],self.last['cameras']['agentview'])
                    if pixel is None: raise ValueError('look_at target behind camera')
                    size=max(32,int(512/k['zoom'])); xy=np.clip(pixel-size/2,0,512-size).astype(int)
                    self.crop=(int(xy[0]),int(xy[1]),int(xy[0]+size),int(xy[1]+size))
                elif n=='done':
                    self.done=True
                elif n=='home':
                    self._move(self.home[:3],np.column_stack([self.home[3:].reshape(2,3).T,np.cross(self.home[3:6],self.home[6:9])]))
                elif n in ('open_gripper','close_gripper'):
                    self._gripper(n=='close_gripper')
                elif n=='goto':
                    r=self._rotation(k['yaw_deg'],k['pitch_deg'],k['azimuth_deg'],k['roll_deg'])
                    if not np.isnan(k['arc_z']):
                        self._move([*self.last['eef_state'][:2],k['arc_z']],speed=k['speed'])
                        self._move([k['x'],k['y'],k['arc_z']],r,k['speed'])
                    self._move([k['x'],k['y'],k['z']],r,k['speed'])
                elif n=='move_by':
                    r=self._rotation()
                    from .upstream import describe_orientation
                    # Upstream yaw describes jaw axis; Panda uses the remapped X axis.
                    desc=describe_orientation(r @ np.array([[0,1,0],[-1,0,0],[0,0,1]]))
                    rot=self._rotation(desc['jaw_yaw_deg']+k['dyaw_deg'],desc['pitch_deg']+k['dpitch_deg'],desc['azimuth_deg'],k['droll_deg'])
                    self._move(np.array(self.last['eef_state'][:3])+[k['forward'],k['left'],k['dz']],rot,k['speed'])
                elif n=='move_until_contact':
                    self._move(np.array(self.last['eef_state'][:3])+[0,0,k['dz']],speed=k['speed'],guard=k['max_lag'])
                    self._move(np.array(self.last['eef_state'][:3])+[0,0,k['relief']],speed=k['speed'],fingers=True)
                elif n in ('grasp','place','place_until_contact','place_relative'):
                    grasp=n=='grasp'
                    if n=='place_relative':
                        if self.grasp_offset is None or not self.holding:
                            raise ValueError('place_relative requires a retained grasp with recorded support height')
                        current=np.asarray(self.last['eef_state'][3:9]).reshape(2,3).T
                        approach=np.cross(current[:,0],current[:,1])
                        if self.grasp_approach is None or abs(approach[2])<.94 or np.dot(approach,self.grasp_approach)<.985:
                            raise ValueError('grasp height offset is invalid after a tilt; use explicit place height')
                        k['z']=k['support_z']+self.grasp_offset+k['release_clearance']
                    r=self._rotation(k['yaw_deg'],k.get('pitch_deg',math.nan),k.get('azimuth_deg',math.nan))
                    if grasp: self._gripper(False)
                    self._move([k['x'],k['y'],k['z']+k['approach_height']],r,k['speed'])
                    descent_z=max(self.last['table_z'],k['z']-.02) if n=='place_until_contact' else k['z']
                    if self.efficient_contact and descent_z+.02<self.last['eef_state'][2]:
                        self._move([k['x'],k['y'],descent_z+.02],r,min(k['speed'],.08),fingers=True)
                    self._move([k['x'],k['y'],descent_z],r,.04,fingers=True,
                               guard=k.get('max_lag') if n=='place_until_contact' else None)
                    if self.done: break
                    if n=='place_until_contact':
                        self._move(np.array(self.last['eef_state'][:3])+[0,0,k['relief']],speed=.04,fingers=True)
                    self._gripper(grasp)
                    if grasp and not self.holding:
                        raise ValueError('grasp_failed')
                    if grasp:
                        support=k.get('support_z',math.nan)
                        self.grasp_offset=(float(self.last['eef_state'][2])-support
                            if np.isfinite(support) else None)
                        columns=np.asarray(self.last['eef_state'][3:9]).reshape(2,3).T
                        self.grasp_approach=np.cross(columns[:,0],columns[:,1])
                    else:
                        self.grasp_offset=None
                    self._move([k['x'],k['y'],k['z']+k['approach_height']],r,
                               min(k['speed'],.08),fingers=True)
                else:
                    raise ValueError('unsupported primitive')
                executed.append(dict(primitive=n,arguments={a:v for a,v in k.items() if isinstance(v,str) or np.isfinite(v)},
                                     holding=self.holding,steps_used=self.steps-start_step))
            status='success' if self.success else 'completed'
        except (ValueError,np.linalg.LinAlgError) as error:
            status=str(error)
        self.feedback=json.dumps(dict(status=status,executed=executed,check=self.last_check,
                                      steps_remaining=max(0,self.max_steps-self.steps)))
        return self.feedback

    def close(self):
        (self.folder/'motion.json').write_text(json.dumps(self.motion_log))
        self.robot.close()

import argparse
import gym
import numpy as np
from constants import params
from gym.envs.mujoco import mujoco_env
import random
from gym import error
import os
import mujoco_py
from collections import OrderedDict
from tempfile import TemporaryFile
from utils.env_utils import convert_observation_to_space
from oscillator import hopf_mod_step
from reward import FitnessFunctionV2
import copy
import xml.etree.ElementTree as ET
import tempfile
import cv2

class Quadruped(gym.GoalEnv, gym.utils.EzPickle):
    def __init__(self,
                 model_path = 'ant.xml',
                 frame_skip = 1,
                 render = False,
                 track_lst = [
                     'desired_goal', 'joint_pos', 'action',
                     'velocity', 'position', 'true_joint_pos',
                     'sensordata', 'qpos', 'qvel',
                     'achieved_goal', 'observation', 'heading_ctrl',
                     'omega', 'z', 'mu',
                     'd1', 'd2', 'd3',
                     'stability', 'omega_o', 'reward'
                 ],
                 obstacles = False,
                 verbose = 0):
        gym.GoalEnv.__init__(self)
        gym.utils.EzPickle.__init__(self)
        self._reward = 0.0
        self._track_lst = track_lst
        self._track_item = {key : [] for key in self._track_lst}
        self._step = 0
        self.verbose = verbose
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.getcwd(), "assets", 'xml', model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self._frame_skip = frame_skip
        self._render_obstacles = obstacles
        if self._render_obstacles:
            tree = ET.parse(fullpath)
            worldbody = tree.find(".//worldbody")
            for i in range(params['num_obstacles']):
                x = np.random.uniform(low = -5.0, high = 5.0)
                y = np.random.uniform(low = -5.0, high = 5.0)
                h = np.random.uniform(low = 0.0, high = params['max_height'])
                if x < 0.2 and x > -0.2:
                    if x > 0:
                        x += 0.2
                    else:
                        x -= 0.2
                if y < 0.2 and y > -0.2:
                    if y > 0:
                        y += 0.2 
                    else:
                        y -+ 0.2
                length = np.random.uniform(low = 0.0, high = params['max_size'])
                width = np.random.uniform(low = 0.0, high = params['max_size'])
                ET.SubElement(
                    worldbody,
                    "geom",
                    name=f"block_{i}",
                    pos=f"{x} {y} {h}",
                    size=f"{length} {width} {h}",
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.4 0.4 0.4 1",
                )
            _, fullpath = tempfile.mkstemp(text=True, suffix=".xml")
            tree.write(fullpath)
            self.worldtree = tree
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        self._viewers = {}
        self._frequency = 2.8
        self.model.opt.timestep = params['dt']
        self._last_base_position = np.array(
            [0, 0, params['INIT_HEIGHT']], dtype = np.float32
        )

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = np.concatenate([
            self._last_base_position, # base position
            np.zeros((4,), dtype = np.float32), # base angular position (quaternion)
            params['INIT_JOINT_POS'] # joint angular position
        ], -1)

        self.init_qvel = np.concatenate([
            np.zeros((3,), dtype = np.float32), # base translational velocity
            np.zeros((3,), dtype = np.float32), # base angular velocity (euler)
            np.zeros(shape = params['INIT_JOINT_POS'].shape, dtype = np.float32) # joint angular velocity
        ], -1)

        self._is_render = render
        self._num_joints = self.init_qpos.shape[-1] - 7
        self._num_legs = params['num_legs']
        self.joint_pos = self.sim.data.qpos[-self._num_joints:]

        self.end_eff = params['end_eff']
        self.support_points = []
        self.times = []
        self.current_supports = []
        self.t = 0
        self.reward = FitnessFunctionV2(params)

        self._set_action_space()
        action = self.action_space.sample()
        self.action = np.zeros(self._action_dim)
        self._track_attr()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()
        self._distance_limit = float("inf")

        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0

    def get_feet_contacts(self):
        contact_points = []
        contact_names = []
        for c in range(self.sim.data.ncon):
            if self.sim.data.contact[c].geom2 in self.end_eff and \
                    self.sim.data.contact[c].geom2 not in contact_names:
                contact_names.append(self.sim.data.contact[c].geom2)
                contact_points.append(self.sim.data.contact[c].pos)
        return list(zip(contact_names, contact_points))

    def set_support_points(self):
        contacts = self.get_feet_contacts()
        self.t += 1
        upright = True
        if len(contacts) > 0:
            upright = True
            if len(contacts) > 2:
                contacts = contacts[:2]
            for c in contacts:
                if c[0] in self.current_supports:
                    index = self.current_supports.index(c[0])
                    if len(self.current_supports) > 1:
                        self.support_points.pop(-2 + index)
                        self.support_points.insert(-2 + index, copy.deepcopy(c[1]))
                        self.times.pop(-2 + index)
                        self.times.insert(-2 + index, copy.deepcopy(self.t))
                        order = [len(self.current_supports) - 1 - index, index]
                        self.current_supports = [self.current_supports[-2 + i] \
                            for i in  order]
                        order = list(range(len(self.support_points)))
                        order[-2 + index] = -2 - index + 1
                        order[-2 - index +1] = -2 + index
                        self.support_points = [self.support_points[i] \
                            for i in  order]
                        self.times = [self.times[i] for i in order]
                    else:
                        if len(self.support_points) > 1:
                            self.support_points.pop(-2 + index)
                            self.support_points.insert(-2 + index, copy.deepcopy(c[1]))
                            self.times.pop(-2 + index)
                            self.times.insert(-2 + index, copy.deepcopy(self.t))
                        else:
                            self.support_points.append(copy.deepcopy(c[1]))
                            self.times.append(copy.deepcopy(self.t))
                else:
                    self.current_supports.append(c[0])
                    self.support_points.append(copy.deepcopy(c[1]))
                    self.times.append(copy.deepcopy(self.t))
                    if len(self.current_supports) > 2:
                        self.current_supports.pop(0)
                    if len(self.support_points) > 6:
                        self.support_points.pop(0)
                    if len(self.times) > 6:
                        self.times.pop(0)

        else:
            upright = False
        return upright

    def calculate_stability_reward(self, d):
        reward = 0.0
        d1 = 0.0
        d2 = 0.0
        d3 = 0.0
        upright = self.set_support_points()
        if len(self.support_points) < 6:
            pass
        else:
            if not upright:
                reward += -2.0
            else:
                Tb = self.times[-1] - self.times[0]
                t = self.times[3] - self.times[1]
                self.reward.build(
                    t, Tb,
                    self.support_points[2],
                    self.support_points[3],
                    self.support_points[0],
                    self.support_points[1],
                    self.support_points[4],
                    self.support_points[5]
                )
                eta = 0
                vd = np.linalg.norm(d[:3])
                if vd != 0:
                    eta = (params['L'] + params['W'])/(2*vd)
                d1, d2, d3, stability = \
                    self.reward.stability_reward(
                        self.sim.data.qpos[:3],
                        self.sim.data.qacc[:3],
                        self.sim.data.qvel[:3],
                        d[3:],
                        eta
                    )
                reward += stability
        return d1, d2, d3, reward, upright

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def set_control_params(self, omega, mu, w, z):
        self._frequency = omega / (2 * np.pi)
        self._amplitude = mu
        self.omega = omega
        self.w = w
        self.z = z

    @property
    def dt(self):
        return self.model.opt.timestep * self._frame_skip

    def _set_leg_params(self):
        """
            modify this according to leg construction
        """
        self.p = 0.01600
        self.q = 0.00000
        self.r = 0.02000
        self.c = 0.01811
        self.u = 0.00000
        self.v = 0.00000
        self.e = -0.06000
        self.h = -0.02820
        self.s = 0.02200
        self.d1 = 0.0
        self.d2 = 0.0
        self.d3 = 0.0
        self.stability = 0.0

    def _set_action_space(self):
        self._set_leg_params()
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = low * params['delta']
        high = high * params['delta']
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _get_init_gamma(self, gait, task, direction):
        init_gamma = [0.0, 0.0, 0.0, 0.0]
        heading_ctrl  = np.ones(4, dtype = np.float32)
        if gait == 'ds_crawl':
            init_gamma = [0.0, 0.5, 0.75, 0.25]
        elif gait == 'ls_crawl':
            init_gamma = [0, 0.5, 0.25, 0.75]
        elif gait == 'trot':
            init_gamma = [0, 0.5, 0, 0.5]
        elif gait == 'pace':
            init_gamma = [0, 0.5, 0.5, 0]
        elif gait == 'bound':
            init_gamma = [0, 0, 0.5, 0.5]
        elif gait == 'transverse_gallop':
            init_gamma = [0, 0.1, 0.6, 0.5]
        elif gait == 'rotary_gallop':
            init_gamma = [0, 0.1, 0.5, 0.6]
        if task == 'straight':
            if direction == 'forward':
                heading_ctrl = np.array([1.0, -1.0, -1.0, 1.0], dtype = np.float32)
            if direction == 'backward':
                init_gamma = init_gamma[-2:] + init_gamma[:2]
                heading_ctrl = np.array([-1.0, 1.0, 1.0, -1.0], dtype = np.float32)
            if direction == 'left':
                init_gamma = init_gamma[1:] + init_gamma[:1]
                heading_ctrl = np.array([1.0, 1.0, -1.0, -1.0], dtype = np.float32)
            if direction == 'right':
                init_gamma = init_gamma[-1:] + init_gamma[:-1]
                heading_ctrl = np.array([-1.0, -1.0, 1.0, 1.0], dtype = np.float32)
        if task == 'rotate':
            if direction == 'left':
                init_gamma = [0.75, 0.5, 0.25, 0.0]
                heading_ctrl = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32)
            if direction == 'right':
                init_gamma = [0, 0.25, 0.5, 0.75]
                heading_ctrl = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32)
        if task == 'turn':
            heading_ctrl = np.array([1.0, -1.0, -1.0, 1.0], dtype = np.float32)
        init_gamma = np.array(init_gamma, dtype = np.float32)
        heading_ctrl *= 1.0
        C = np.load('assets/out/plots/coef.npy')
        return init_gamma, heading_ctrl

    def reset(self):
        self._step = 0
        self._last_base_position = np.array(
            [0, 0, params['INIT_HEIGHT']], dtype = np.float32
        )
        self.gamma = self.init_gamma.copy()
        #self.z = np.concatenate([np.cos((self.init_gamma + params['offset']) * np.pi * 2), np.sin((self.init_gamma + params['offset']) * np.pi * 2)], -1)
        self.z = self._get_z()
        self.sim.reset()
        self.ob = self.reset_model()
        if len(self._track_lst) > 0 and self.verbose > 0:
            for item in self._track_lst:
                with open(os.path.join('assets', 'episode','ant_{}.npy'.format(item)), 'wb') as f:
                    np.save(f, np.stack(self._track_item[item], axis = 0))
        raise NotImplementedError
        self.d1, self.d2, self.d3, self.stability, upright = self.calculate_stability_reward(self.desired_goal)
        self._reset_track_lst()
        self._track_attr()
        return self.ob

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
            modify this according to observation space
        """
        ob = {}

        q = self.sim.data.qpos[-self._num_joints:]
        q_dot = self.sim.data.qvel[-self._num_joints:]
        R = self.sim.data.qpos[3:7]
        w_b = self.sim.data.qpos[3:6]

        v_b = self.sim.data.qvel[:3]
        frame_b = self.sim.data.qpos[:4]

        I_v_b = v_b
        b_v_b = v_b

        ob = 
        
        return ob

    def step(self, action, callback=None):
        #action = np.clip(action, -1, 1) # modify this according to appropriate bounds
        self.joint_pos += action
        self.sim.data.ctrl[:] = self.joint_pos
        for _ in range(0, self.frame_skip):
            self.sim.step()
        ob = self._get_obs()
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _leg_kinematics(self, knee_pos):
        """
            modify this according to leg construction
        """
        knee_pos = np.abs(knee_pos)
        sign = np.sign(knee_pos)
        t = np.sqrt(self.p ** 2 + self.q ** 2 + self.r ** 2 - 2 * np.sqrt(self.p ** 2 + self.q ** 2) * self.r * np.cos(knee_pos))
        phi = np.arccos((self.c ** 2 + t ** 2 - self.s **2) / (2 * self.c * t))
        delta = np.arcsin((self.c * np.sin(phi)) / self.s)
        beta = np.arcsin((self.r * np.sin(knee_pos)) / t)
        epsilon = np.pi - (delta + beta)
        Ax = self.r * np.cos(knee_pos) + self.u
        Ay = self.r * np.sin(knee_pos) + self.v
        Bx = self.s * np.cos(epsilon) + self.u + self.p
        By = self.s * np.cos(epsilon) + self.v + self.q
        Cx = Ax + ((Bx - Ax) * self.e + (Ay - By) * self.h) / self.c
        Cy = Ay + ((By - Ay) * self.e + (Ax - Bx) * self.h) / self.c
        alpha = np.arctan((Cy - Ay) / (Cx - Ax))
        return alpha * sign

    def _track_attr(self):
        """
            modify this according to need
        """
        self._track_item['joint_pos'].append(self.joint_pos.copy())
        self._track_item['action'].append(self.action.copy())
        self._track_item['velocity'].append(self.sim.data.qvel[:6].copy())
        self._track_item['position'].append(self.sim.data.qpos[:3].copy())
        self._track_item['true_joint_pos'].append(self.sim.data.qpos[-self._num_joints:].copy())
        self._track_item['sensordata'].append(self.sim.data.sensordata.copy())
        self._track_item['qpos'].append(self.sim.data.qpos.copy())
        self._track_item['qvel'].append(self.sim.data.qvel.copy())
        ob =  self._get_obs()
        self._track_item['achieved_goal'].append(ob['achieved_goal'].copy())
        self._track_item['desired_goal'].append(ob['desired_goal'].copy())
        self._track_item['observation'].append(ob['observation'].copy())
        self._track_item['heading_ctrl'].append(self.heading_ctrl.copy())
        self._track_item['omega_o'].append(self.omega.copy())
        self._track_item['omega'].append(self.w.copy())
        self._track_item['z'].append(self.z.copy())
        self._track_item['mu'].append(self.mu.copy())
        self._track_item['d1'].append(np.array([self.d1], dtype = np.float32))
        self._track_item['d2'].append(np.array([self.d2], dtype = np.float32))
        self._track_item['d3'].append(np.array([self.d3], dtype = np.float32))
        self._track_item['stability'].append(np.array([self.stability], dtype = np.float32))
        self._track_item['reward'].append(np.array([self._reward], dtype = np.float32))

    def _get_track_item(self, item):
        return self._track_item[item].copy()

    def _reset_track_lst(self):
        """
             modify this according to need
        """
        del self._track_item
        self._track_item = {key : [] for key in self._track_lst}
        return self._track_item

    def render(self,
               mode='human',
               width=params['DEFAULT_SIZE'],
               height=params['DEFAULT_SIZE'],
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

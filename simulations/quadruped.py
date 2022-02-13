from simulations.agent_model import AgentModel
from constants import params
from typing import Any, List, Optional, Tuple, Type
from reward import FitnessFunctionV2

class Quadruped(AgentModel):
    
    VELOCITY_LIMITS: float = 4.0

    def __init__(self,
        file_path: Optional[str] = 'ant.xml',
        verbose: Optional[int] = 0,
        track_lst: Optional[List[str]] = [
                     'desired_goal', 'joint_pos', 'action',
                     'velocity', 'position', 'true_joint_pos',
                     'sensordata', 'qpos', 'qvel',
                     'achieved_goal', 'observation', 'heading_ctrl',
                     'omega', 'z', 'mu',
                     'd1', 'd2', 'd3',
                     'stability', 'omega_o', 'reward'
                 ],
        frame_skip: Optional[int] = 1,
        render_obstacles: Optional[bool] = False,
    ) -> None:
        file_path = os.path.join(
            os.getcwd(),
            'assets',
            'xml',
            file_path
        )

        self._track_lst = track_lst
        self._track_item = {key : [] for key in self._track_lst}

        self._last_base_position = np.array(
            [0, 0, params['INIT_HEIGHT']], dtype = np.float32
        )
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

        self._num_joints = self.init_qpos.shape[-1] - 7
        self._num_legs = params['num_legs']
        self.joint_pos = self.sim.data.qpos[-self._num_joints:]

        self._frequency = 0.0
        self._amplitude = 0.0
        self._reward = 0.0

        self._n_steps = 0
        self._step = 0
        self.verbose = verbose

        self._update_action_every = params['update_action_every']

        self._frame_skip = frame_skip
        
        self._render_obstacles = render_obstacles
        if self._render_obstacles:
            tree = ET.parse(file_path)
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
            _, file_path = tempfile.mkstemp(text=True, suffix=".xml")

        self.end_eff = params['end_eff']

        self.support_points = []
        self.times = []
        self.current_supports = []
        self.t = 0

        self.reward = FitnessFunctionV2(params)  
        super().__init__(file_path, self._frame_skip)

    def _reset_track_lst(self):
        """ 
             modify this according to need
        """
        del self._track_item
        self._track_item = {key : [] for key in self._track_lst}
        return self._track_item

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

    def _set_action_space(self):
        """
            Need to modify to be CPG parameter space and connection configuration
        """
        low = np.array([0.0, -np.pi], dtype = np.float32)
        high = np.array([self.VELOCITY_LIMITS * 1.41, np.pi], dtype = np.float32)
        self.action_dim = 2 
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_obs(self):
        rgb, depth = self.sim.render(
            width = 100,
            height = 75,
            camera_name = 'mtdcam',
            depth = True
        )
        depth = (depth - 0.92) / 0.08
        img = np.flipud(np.concatenate([
            rgb, np.expand_dims(depth, -1)
        ], -1))
        obs = {
            'front' : np.flipud(img),
            'front_depth' : np.flipud(depth)
        }
        return obs

    def step(self, action, callback=None):
        action = np.clip(action, -1, 1) # modify this according to appropriate bounds
        reward, done, info = self.do_simulation(action, n_frames = self._frame_skip)
        ob = self._get_obs()
        return ob, reward, done, info

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
        if self.policy_type == 'MultiInputPolicy':
            """
                modify this according to observation space
            """
            self.achieved_goal = self.sim.data.qvel[:6].copy()
            self.command = random.choice(self.commands)
            if self.verbose > 0:
                print('[Quadruped] Command is `{}` with gait `{}` in task `{}` and direction `{}`'.format(self.command, self.gait, self.task, self.direction))
            self.desired_goal = self.command.copy()

        if len(self._track_lst) > 0 and self.verbose > 0:
            for item in self._track_lst:
                with open(os.path.join('assets', 'episode','ant_{}.npy'.format(item)), 'wb') as f:
                    np.save(f, np.stack(self._track_item[item], axis = 0))
        self.d1, self.d2, self.d3, self.stability, upright = self.calculate_stability_reward(self.desired_goal)
        self._reset_track_lst()
        self._track_attr()
        return self.ob

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def do_simulation(self, action, n_frames, callback = None):
        # 4 dimensional action space for left legs and right legs respectively
        self._frequency = np.array([action[0], action[2]], dtype = np.float32)
        self._amplitude = np.array([action[1], action[3]], dtype = np.float32)
        
        omega = 2 * np.pi * self._frequency
        timer_omega = omega[0]
        self.action = action
        counter = 0

        if self.verbose > 0:
            print('Number of Steps: {}'.format(self._n_steps))

        while(np.abs(phase) <= np.pi * self._update_action_every):
            self.joint_pos, timer_omega = self._get_joint_pos(self._amplitude, omega)
            
            posbefore = self.get_body_com("torso").copy()
            penalty = 0.0
            if np.isnan(self.joint_pos).any():
                self.joint_pos = np.nan_to_num(self.joint_pos)
                penalty += -1.0

            self.sim.data.ctrl[:] = self.joint_pos
            for _ in range(n_frames):
                self.sim.step()

            posafter = self.get_body_com("torso").copy()

            velocity = (posafter - posbefore) / self.dt
            ang_vel = self.sim.data.qvel[3:6]

            self.d1, self.d2, self.d3, self.stability, upright = self.calculate_stability_reward(self.desired_goal)
        
            if self.policy_type == 'MultiInputPolicy':
                """ 
                    modify this according to observation space
                """
                if len(self._track_item['achieved_goal']) \
                        > params['window_size']:
                    self.achieved_goal = sum([np.concatenate([
                        velocity,
                        ang_vel
                    ], -1)] + self._track_item[
                        'achieved_goal'
                        ][-params['window_size'] + 1:] 
                    ) / params['window_size']
                else:
                    self.achieved_goal = sum([np.concatenate([
                        velocity,
                        ang_vel
                    ], -1)] + [self._track_item[
                        'achieved_goal'][0]] * (params['window_size'] - 1)
                    ) / params['window_size']
            if self._is_render:
                pass
                #self.render()
            if self.policy_type == 'MultiInputPolicy':
                reward_velocity += np.linalg.norm(
                    self.achieved_goal - self.desired_goal
                )   
            else:
                reward_velocity += np.abs(
                    self.achieved_goal[0] - self.desired_goal[0]
                )   
            reward_energy += -np.linalg.norm(
                self.sim.data.actuator_force * self.sim.data.qvel[-self._num_joints:]
            ) - np.linalg.norm(np.clip(self.sim.data.cfrc_ext, -1, 1).flat)
            if not upright:
                done = True
            counter += 1
            phase += timer_omega * self.dt * counter
            self._track_attr()
            self._step += 1
            if self._step % params['max_step_length'] == 0:
                break

        self._n_steps += 1
        reward_distance = 1 - np.exp(-np.linalg.norm(self.sim.data.qpos[:2]))
        reward_velocity = np.exp(params['reward_velocity_coef'] * reward_velocity)
        reward_energy = np.exp(params['reward_energy_coef'] * reward_energy)
        reward = reward_distance + reward_velocity + reward_energy + penalty
        self.rewards = np.array([
            reward_velocity,
            reward_distance,
            reward_energy,
            penalty
        ], dtype = np.float32)
        info = { 
            'reward_velocity' : reward_velocity,
            'reward_distance' : reward_distance,
            'reward_energy' : reward_energy,
            'reward' : reward,
            'penalty' : penalty
        }

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.02 and state[2] <= 0.5
        done = not notdone
        self._reward = reward
        if self._step >= params['max_episode_size']:
            done = True
        return reward, done, info

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def calculate_stability_reward(self, desired_goal):
        """
            Needs desired velocity for stability reward
        """
        raise NotImplementedError

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

    def seed(self, seed=None):
        if seed is None:
            seed = params['seed']
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_feet_contacts(self):
        """
            Refer to the following link for flowchart for this method:
            https://www.notion.so/shandilya1998/Omnidirectional-Controller-Reorganisation-bcbbda8cdf7b4423825e33d115eaa777#9d7e06886b4549a9b76192cac5d5b5a6
        """
        contact_points = []
        contact_names = []
        for c in range(self.sim.data.ncon):
            if self.sim.data.contact[c].geom2 in self.end_eff and \
                    self.sim.data.contact[c].geom2 not in contact_names:
                contact_names.append(self.sim.data.contact[c].geom2)
                contact_points.append(self.sim.data.contact[c].pos)
        return list(zip(contact_names, contact_points))

    def set_support_points(self):
        """
            Refer to the following link for flowchart for this method:
            https://www.notion.so/shandilya1998/Omnidirectional-Controller-Reorganisation-bcbbda8cdf7b4423825e33d115eaa777#562a8e8f8be24051a12b1410668c66df
        """
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

    def _create_command_lst(self):
        self.commands = []
        xvel = np.zeros((50,))
        yvel = np.zeros((50,))
        zvel = np.zeros((50,))
        roll_rate = np.zeros((50,))
        pitch_rate = np.zeros((50,))
        yaw_rate = np.zeros((50,))
        if self.gait in [
            'ds_crawl',
            'ls_crawl',
            'trot',
            'pace',
            'bound',
            'transverse_gallop',
            'rotary_gallop',
        ]:
            if self.task == 'rotate' or self.task == 'turn':
                if self.direction == 'left':
                    yaw_rate = np.random.uniform(
                        low = -0.1, high = -0.001, size = (50,)
                    )
                elif self.direction == 'right':
                    yaw_rate = np.random.uniform(
                        low = 0.001, high = 0.1, size = (50,)
                    )
            elif self.task == 'straight':
                if self.direction == 'left':
                    yvel = np.random.uniform(
                        low = -0.1, high = -0.001, size = (50,)
                    )
                elif self.direction == 'right':
                    yvel = np.random.uniform(
                        low = 0.001, high = 0.1, size = (50,)
                    )
                elif self.direction == 'forward':
                    xvel = np.random.uniform(
                        low = 0.001, high = 0.1, size = (50,)
                    )
                elif self.direction == 'backward':
                    xvel = np.random.uniform(
                        low = -0.1, high = -0.001, size = (50,)
                    )
            else:
                raise ValueError

        self.commands = np.stack(
            [yvel, xvel, zvel, roll_rate, pitch_rate, yaw_rate], -1
        )

        return list(self.commands)

    def _get_track_item(self, item):
        return self._track_item[item].copy()

    def _get_joint_pos(self, mu, omega):
        """
            CPG model driving locomotion
        """
        out = []
        amp = []
        omg = []
        if self._action_dim == 2:
            amp.extend([mu[0]] * 4)
            omg.extend([omega[0]]* 4)
        elif self._action_dim == 4:
            amp.extend([mu[0], mu[1], mu[1], mu[0]])
            omg.extend([omega[0], omega[1], omega[1], omega[0]])
        time_omega = 0.0
        self.mu = np.array(amp, dtype = np.float32)
        self.omega = np.array(omg, dtype = np.float32) * self.heading_ctrl
        self.z, w = hopf_mod_step(self.omega, self.mu, self.z, self.C,
                params['degree'], self.dt)
        self.w = w
        out = []
        for i in range(self._num_legs):
            direction = 1.0
            if i in [0, 3]:
                direction = 1.0
            if i in [1,2]:
                direction = -1.0
            out.append(self.z[i] * np.tanh(1e3 * self.omega[i]))
            knee = -np.maximum(-self.z[self._num_legs + i], 0)
            out.append(knee * direction)
            out.append((-0.35 * knee  + 1.3089) * direction)
        out = np.array(out, dtype = np.float32)
        return out, w.max()

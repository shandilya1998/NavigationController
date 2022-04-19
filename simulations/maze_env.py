#!/usr/bin/env python3
"""
Mujoco Maze environment.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

"""
    REFER TO THE FOLLOWING FOR GEOM AND BODY NAMES IN MODEL:
        https://github.com/openai/mujoco-py/blob/9dd6d3e8263ba42bfd9499a988b36abc6b8954e9/mujoco_py/generated/wrappers.pxi
"""

import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple, Type
import gym
from networkx.algorithms.dag import transitive_closure
import numpy as np
import networkx as nx
from simulations import maze_env_utils, maze_task
from simulations.agent_model import AgentModel
from utils.env_utils import convert_observation_to_space, \
    calc_spline_course, TargetCourse, proportional_control, \
    State, pure_pursuit_steer_control
import random
import copy
from constants import params, image_width, image_height
import math
import cv2
import colorsys
from simulations.maze_task import Rgb
import open3d as o3d
from utils.point_cloud import rotMatList2NPRotMat, quat2Mat, posRotMat2Mat, \
    point_cloud_2_birdseye
import matplotlib.pyplot as plt

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.join(os.getcwd(), 'assets', 'xml')


class MazeEnv(gym.Env):
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
        max_episode_size: int = 2000,
        n_steps = 50,
        include_position: bool = True,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        inner_reward_scaling: float = 1.0,
        restitution_coef: float = 0.8,
        task_kwargs: dict = {},
        websock_port: Optional[int] = None,
        camera_move_x: Optional[float] = None,
        camera_move_y: Optional[float] = None,
        camera_zoom: Optional[float] = None,
        image_shape: Tuple[int, int] = (600, 480),
        mode = None,
        **kwargs,
    ) -> None:
        self.mode = mode
        self.collision_count = 0
        self.n_steps = n_steps
        self.kwargs = kwargs
        self.top_view_size = params['top_view_size']
        self.t = 0  # time steps
        self.total_steps = 0 
        self.ep = 0
        self.max_episode_size = max_episode_size
        self._task = maze_task(maze_size_scaling, **task_kwargs)
        self._maze_height = height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self._inner_reward_scaling = inner_reward_scaling
        self._put_spin_near_agent = self._task.PUT_SPIN_NEAR_AGENT
        # Observe other objectives
        self._restitution_coef = restitution_coef

        self._maze_structure = structure = self._task.create_maze()
        self._open_position_indices = []
        for i in range(len(self._maze_structure)):
            for j in range(len(self._maze_structure[i])):
                if not self._maze_structure[i][j].is_wall_or_chasm():
                    self._open_position_indices.append([i, j])

        # Elevate the maze to allow for falling.
        self.elevated = any(maze_env_utils.MazeCell.CHASM in row for row in structure)
        # Are there any movable blocks?
        self.blocks = any(any(r.can_move() for r in row) for row in structure)
        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self._init_positions = [
            (x - torso_x, y - torso_y) for x, y in self._find_all_robots()
        ]

        def func(x):
            x_int, x_frac = int(x), x % 1
            if x_frac > 0.5:
                x_int += 1
            return x_int

        def func2(x):
            x_int, x_frac = int(x), x % 1 
            if x_frac > 0.5:
                x_int += 1
                x_frac -= 0.5
            else:
                x_frac += 0.5
            return x_int, x_frac

        self._xy_to_rowcol = lambda x, y: (
            func((y + torso_y) / size_scaling),
            func((x + torso_x) / size_scaling),
        )
        self._xy_to_rowcol_v2 = lambda x, y: (
            func2((y + torso_y) / size_scaling),
            func2((x + torso_x) / size_scaling), 
        )
        self._rowcol_to_xy = lambda r, c: (
            c * size_scaling - torso_y,
            r * size_scaling - torso_x
        )

        # Let's create MuJoCo XML
        self.model_cls = model_cls
        self._websock_port = websock_port
        self._camera_move_x = camera_move_x
        self._camera_move_y = camera_move_y
        self._camera_zoom = camera_zoom
        self._image_shape = image_shape
        self._mj_offscreen_viewer = None
        self._websock_server_pipe = None
        self.set_env()

    def _ensure_distance_from_target(self, row, row_frac, col, col_frac, pos):
        target_pos = self._task.goals[self._task.goal_index].pos
        (pos_row, _), (pos_col, _) = self._xy_to_rowcol_v2(target_pos[0], target_pos[1])
        if [pos_row, pos_col] == [row, col]:
            row, col = random.choice(self._open_position_indices)
            row_frac = np.random.uniform(low = -0.4, high = 0.4)
            col_frac = np.random.uniform(low = -0.4, high = 0.4)
            pos = self._rowcol_to_xy(row + row_frac, col + col_frac)
            (row, row_frac), (col, col_frac), pos = self._ensure_distance_from_target(
                row, row_frac, col, col_frac, pos
            )
        return (row, row_frac), (col, col_frac), pos

    def _set_init(self, agent):
        row, col = agent
        row_frac = np.random.uniform(low = -0.4, high = 0.4)
        col_frac = np.random.uniform(low = -0.4, high = 0.4)
        pos = self._rowcol_to_xy(row + row_frac, col + col_frac)
        self._start_pos = np.array(pos, dtype = np.float32)
        (row, row_frac), (col, col_frac), pos = self._ensure_distance_from_target(
            row, row_frac, col, col_frac, pos
        )
        struct = self._maze_structure[row][col]
        if struct.is_block():
            neighbors = [
                [row + 1, col],
                [row, col + 1],
                [row - 1, col],
                [row, col - 1],
                [row + 1, col + 1],
                [row - 1, col + 1],
                [row + 1, col - 1],
                [row - 1, col - 1]
            ]
            eligible = []
            for neighbor in neighbors:
                r, c = neighbor
                if self._check_structure_index_validity(r, c):
                    if not self._maze_structure[r][c].is_block():
                        eligible.append([r, c])
            choice = random.choice(eligible)
            d_r = choice[0] - row
            d_c = choice[1] - col
            row = r + d_r + 1
            col = c + d_c + 1
            x, y = self._rowcol_to_xy(row, col)
            pos = np.array([x, y], dtype = np.float32)

        (row, row_frac), (col, col_frac) = self._xy_to_rowcol_v2(pos[0], pos[1])
        neighbors = [
            [row + 1, col],
            [row, col + 1],
            [row - 1, col],
            [row, col - 1],
            [row + 1, col + 1],
            [row - 1, col + 1],
            [row + 1, col - 1],
            [row - 1, col - 1]
        ]

        possibilities = [-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 
            0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        for neighbor in neighbors:
            r, c = neighbor
            if self._check_structure_index_validity(r, c):
                if self._maze_structure[r][c].is_block():
                    x, y = self._rowcol_to_xy(row, col)
                    _x, _y = self._rowcol_to_xy(r, c)
                    angle = np.arctan2(_y - y, _x - x)
                    index = possibilities.index(angle)
                    possibilities.pop(index)

        ori = np.random.choice([
                np.random.uniform(low = p - np.pi / 4, high = p + np.pi / 4) for p in possibilities
        ])

        if ori > np.pi:
            ori = ori - 2 * np.pi
        elif ori < -np.pi:
            ori = ori + 2 * np.pi

        return pos, ori

    def set_env(self):
        xml_path = os.path.join(MODEL_DIR, self.model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        height_offset = 0.0
        if self.elevated:
            # Increase initial z-pos of ant.
            height_offset = self._maze_height * self._maze_size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set("pos", f"0 0 {0.75 + height_offset:.2f}")
        if self.blocks:
            # If there are movable blocks, change simulation settings to perform
            # better contact detection.
            default = tree.find(".//default")
            default.find(".//geom").set("solimp", ".995 .995 .01")

        tree.find('.//option').set('timestep', str(params['dt']))
        self.movable_blocks = []
        self.object_balls = []
        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self.obstacles = []
        for i in range(len(self._maze_structure)):
            for j in range(len(self._maze_structure[0])):
                struct = self._maze_structure[i][j]
                if struct.is_robot() and self._put_spin_near_agent:
                    struct = maze_env_utils.MazeCell.SPIN
                x, y = j * self._maze_size_scaling - torso_x, i * self._maze_size_scaling - torso_y
                h = self._maze_height / 2 * self._maze_size_scaling
                size = self._maze_size_scaling * 0.5
                if self.elevated and not struct.is_chasm():
                    rgba = "0 0 0 1"
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"elevated_{i}_{j}",
                        pos=f"{x} {y} {h}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="MatObj",
                        contype="1",
                        conaffinity="1",
                        rgba=rgba
                    )
                    self.obstacles.append(f"elevated_{i}_{j}")
                if struct.is_block():
                    # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    rgba = "0 0 0 1"
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {h + height_offset}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="MatObj",
                        contype="1",
                        conaffinity="1",
                        rgba=rgba,
                    )
                    self.obstacles.append(f"block_{i}_{j}")
                elif struct.can_move():
                    # Movable block.
                    self.movable_blocks.append(f"movable_{i}_{j}")
                    _add_movable_block(
                        worldbody,
                        struct,
                        i,
                        j,
                        self._maze_size_scaling,
                        x,
                        y,
                        h,
                        height_offset,
                    )
                    self.obstacles.append(f"movable_{i}_{j}")
                elif struct.is_object_ball():
                    # Movable Ball
                    self.object_balls.append(f"objball_{i}_{j}")
                    _add_object_ball(worldbody, i, j, x, y, self._task.OBJECT_BALL_SIZE)
                    self.obstacles.append(f"objball_{i}_{j}")

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if "name" not in geom.attrib:
                raise Exception("Every geom of the torso must have a name")

        # Set goals
        sampled_cells = random.sample(self._open_position_indices, 15)
        agent = copy.deepcopy(sampled_cells[-1])
        goals = copy.deepcopy(sampled_cells[:-1])
        target_index = np.random.randint(0, len(goals))
        target_rgb = [0.7, 0.1, 0.1]
        if self.mode == 'vae':
            target_rgb = random.sample([
                np.random.uniform(low = 0.1, high = 0.5),
                np.random.uniform(low = 0.1, high = 0.5),
                np.random.uniform(low = 0.5, high = 1),
            ], k = 3)
        target_hsv = colorsys.rgb_to_hsv(*target_rgb)
        available_h = []
        h = target_hsv[0] * 180
        if h < 50:
            available_h.append([h * 180 + 50, 180 - (50 - h)])
        elif h < 130:
            available_h.append([0, h - 50])
            available_h.append([h + 50, 180])
        else:
            available_h.append([180 - h, h - 50])

        sample_h = lambda: np.random.uniform(low = available_h[0][0], high = available_h[0][1])
        if len(available_h) == 2:
            prob = np.array([
                available_h[0][1] - available_h[0][0],
                available_h[1][1] - available_h[1][0]
            ])
            prob = prob / prob.sum()
            sample_h = lambda: np.random.choice([
                np.random.uniform(low = available_h[0][0], high = available_h[0][1]),
                np.random.uniform(low = available_h[1][0], high = available_h[1][1])
            ], p = prob)

        for i, goal in enumerate(goals):
            site_type = random.choice(['capsule', 'ellipsoid', 'cylinder', 'box', 'sphere'])
            r, g, b = 0, 0, 0
            h, s, v = 0, 0, 0
            rgb = None
            if target_index == i:
                rgb = Rgb(*target_rgb)
                h, s, v = copy.deepcopy(target_hsv)
            else:
                h = sample_h()
                if h < 94:
                    s = np.random.uniform(low = 0, high = 255)
                    v = np.random.uniform(low = 51, high = 255)
                else:
                    s = np.random.uniform(low = 80, high = 255)
                    v = np.random.uniform(low = 51, high = 255)
                h = h / 180
                s = s / 255
                v = v / 255
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                rgb = Rgb(r, g, b)
            size = self._maze_size_scaling * 0.25
            if site_type != 'sphere':
                size = np.random.uniform(low = size / 3, high = size, size = (3,)).tolist()
            else:
                size = [size, size, size]
            hsv_low = []
            hsv_high = []
            if h > 10 / 180:
                hsv_low.append(h * 180 - 10)
            else:
                hsv_low.append(0)
            if h < 160 / 180:
                hsv_high.append(h * 180 + 10)
            else:
                hsv_high.append(180)
            if s > 100 / 255:
                hsv_low.append(s * 255 - 100)
            else:
                hsv_low.append(0)
            if s < 155 / 255:
                hsv_high.append(s * 255 + 100)
            else:
                hsv_high.append(255)
            hsv_low.append(0)
            hsv_high.append(255)
            """
            if target_index == i:
                print('hsv range', hsv_low, hsv_high)
                print('hsv value', h * 180, s * 255, v * 255)
                print('rgb, variables', r, g, b)
                print('target rgb', target_rgb)
                print('target_hsv', target_hsv[0] * 180, target_hsv[1] * 255, target_hsv[2] * 255)
                print('rgb', rgb)
            """
            goal.append({
                'hsv_low' : copy.deepcopy(hsv_low),
                'hsv_high' : copy.deepcopy(hsv_high),
                'threshold' : 2.25 if i == target_index else 1.5,
                'target' : True if i == target_index else False,
                'rgb' : copy.deepcopy(rgb),
                'size' : copy.deepcopy(size),
                'site_type' : site_type
            })
        self._task.set(goals, (self._init_torso_x, self._init_torso_y))
        for i, goal in enumerate(self._task.goals):
            z = goal.pos[2] if goal.dim >= 3 else 0.1 *  self._maze_size_scaling
            if goal.custom_size is None:
                size = f"{self._maze_size_scaling * 0.1}"
            else:
                if isinstance(goal.custom_size, list):
                    size = ' '.join(map(str, goal.custom_size))
                else:
                    size = f"{goal.custom_size}"
            """
            if i == self._task.goal_index:
                print(goal.rgb.red, goal.rgb.blue, goal.rgb.green)
                print(self._task.colors[i])
            """
            ET.SubElement(
                worldbody,
                "site",
                name=f"goal_site{i}",
                pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                size=size,
                rgba='{} {} {} 1'.format(goal.rgb.red, goal.rgb.green, goal.rgb.blue),
                material = "MatObj",
                type=f"{goal.site_type}",
            )
        
        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)
        self.world_tree = tree
        self.wrapped_env = self.model_cls(file_path=file_path, **self.kwargs)
        self.model = self.wrapped_env.model
        self.data = self.wrapped_env.data
        self.sim = self.wrapped_env.sim
        self.cam_names = list(self.model.camera_names)
        index = self.cam_names.index('mtdcam1')
        self.cam_body_id = self.sim.model.cam_bodyid[index]
        fovy = math.radians(self.model.cam_fovy[index])
        f = image_height / (2 * math.tan(fovy / 2))
        assert image_height == image_width
        cx = image_width / 2
        cy = image_height / 2
        self.cam_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype = np.float32) 
        self.indices_x = np.repeat(
            np.expand_dims(np.arange(image_height), 1),
            image_width, 1
        ).reshape(-1)
        self.indices_y = np.repeat(
            np.expand_dims(np.arange(image_width), 0),
            image_height, 0
        ).reshape(-1)
        self.pcd = o3d.geometry.PointCloud()
        self.vec = o3d.utility.Vector3dVector()
        self.cam_pos = self.model.body_pos[self.cam_body_id]
        mat = rotMatList2NPRotMat(self.sim.model.cam_mat0[index])
        rot_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        mat = np.dot(mat, rot_mat)
        ext = np.eye(4)
        ext[:3, :3] = mat
        ext[:3, 3] = self.cam_pos
        self.ext = ext
        min_bound = [-35, -35, -0.4]
        max_bound = [35, 35, 1]
        self.pc_target_bounds =  np.array([min_bound, max_bound], dtype = np.float32)

        self._init_pos, self._init_ori = self._set_init(agent)
        self.wrapped_env.set_xy(self._init_pos)
        self.wrapped_env.set_ori(self._init_ori)
        self.dt = self.wrapped_env.dt
        assert self.dt == params['dt']
        self.obstacles_ids = []
        self.agent_ids = []
        for name in self.model.geom_names:
            if 'block' in name:
                self.obstacles_ids.append(self.model._geom_name2id[name])
            elif 'obj' in name:
                self.obstacles_ids.append(self.model._geom_name2id[name])
            elif name != 'floor':
                self.agent_ids.append(self.model._geom_name2id[name])
        self._set_action_space()
        self.last_wrapped_obs = self.wrapped_env._get_obs().copy()
        action = self.action_space.sample()
        self.actions = [np.zeros_like(action) for _ in range(self.n_steps)]
        self.set_goal_path()

    def set_goal_path(self):
        goal = self._task.goals[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals = [goal.copy() for _ in range(self.n_steps)]
        self.positions = [np.zeros_like(self.data.qpos) for _ in range(self.n_steps)]
        self._create_maze_graph()
        self.sampled_path = self._sample_path()
        self._current_cell = copy.deepcopy(self.sampled_path[0])
        self._find_all_waypoints()
        self._find_cubic_spline_path()
        self._setup_vel_control()
        ob = self._get_obs()
        self._set_observation_space(ob)

    def _find_all_waypoints(self):
        self.wx = []
        self.wy = []
        cells = []
        for i in range(len(self.sampled_path)):
            row, col = self._graph_to_structure_index(self.sampled_path[i])
            cells.append([row, col])
            if i < len(self.sampled_path) - 1:
                n_row, n_col = self._graph_to_structure_index(self.sampled_path[i + 1])
                d_r = n_row - row
                d_c = n_col - col
                if abs(d_r) > 0 and abs(d_c) > 0:
                    if self._maze_structure[row + d_r][col].is_block():
                        cells.append([row, col + d_c])
                    elif self._maze_structure[row][col + d_c].is_block():
                        cells.append([row + d_r, col])


        for row, col in cells:
            x, y = self._rowcol_to_xy(row, col)
            self.wx.append(copy.deepcopy(x))
            self.wy.append(copy.deepcopy(y))
        self.wx.pop()
        self.wy.pop()
        self.wx.append(self._task.goals[self._task.goal_index].pos[0])
        self.wy.append(self._task.goals[self._task.goal_index].pos[1])
        self.final = [self.wx[-1], self.wy[-1]]

    def _find_cubic_spline_path(self):
        self.cx, self.cy, self.cyaw, self.ck, self.s = calc_spline_course(self.wx, self.wy, params['ds'])

    @property
    def action_space(self):
        return self._action_space

    def _setup_vel_control(self):
        self.target_speed = 2
        self.state = State(
            x = self.wrapped_env.sim.data.qpos[0],
            y = self.wrapped_env.sim.data.qpos[1],
            yaw = self.wrapped_env.sim.data.qpos[2],
            v = np.linalg.norm(self.wrapped_env.sim.data.qvel[:2]),
            WB = 0.2 * self._maze_size_scaling,
        )
        self.last_idx = len(self.cx) - 1
        self.target_course = TargetCourse(self.cx, self.cy)
        self.target_ind, _ = self.target_course.search_target_index(self.state)

    def _sample_path(self):
        robot_x, robot_y = self.wrapped_env.get_xy()
        row, col = self._xy_to_rowcol(robot_x, robot_y)
        source = self._structure_to_graph_index(row, col)
        goal_pos = self._task.goals[self._task.goal_index].pos[:2]
        row, col = self._xy_to_rowcol(goal_pos[0], goal_pos[1])
        target = self._structure_to_graph_index(row, col)
        paths = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(
            self._maze_graph,
            source,
            target
        ))
        return paths[0]

    def get_action(self):
        ai = proportional_control(self.target_speed, self.state.v)
        di, self.target_ind = pure_pursuit_steer_control(
            self.state, self.target_course, self.target_ind
        )
        #yaw = self.state.yaw +  self.state.v / self.state.WB * math.tan(di) * self.dt
        v = self.state.v + ai * self.dt
        vyaw = self.state.v / self.state.WB * math.tan(di)
        #self.state.update(ai, di, self.dt)
        #v = self.state.v
        #yaw = self.state.yaw
        # Refer to simulations/point PointEnv: def step() for more information
        yaw = self.check_angle(self.state.yaw + vyaw * self.dt)
        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)
        self.sampled_action = np.array([
            vyaw,
        ], dtype = np.float32)
        return self.sampled_action

    def _graph_to_structure_index(self, index):
        row = int(index / len(self._maze_structure))
        col = index % len(self._maze_structure[0])
        return row, col

    def _structure_to_graph_index(self, row, col):
        return row * len(self._maze_structure[0]) + col

    def _check_structure_index_validity(self, i, j):
        valid = [True, True]
        if i < 0:
            valid[0] = False
        elif i >= len(self._maze_structure):
            valid[0] = False
        if j < 0:
            valid[1] = False
        elif j >= len(self._maze_structure[0]):
            valid[1] = False
        return valid[0] and valid[1]

    def _add_edges_to_maze_graph(self, node):
        neighbors = [
            (node['row'] - 1, node['col']),
            (node['row'], node['col'] - 1),
            (node['row'] + 1, node['col']),
            (node['row'], node['col'] + 1),
            (node['row'] + 1, node['col'] + 1),
            (node['row'] + 1, node['col'] - 1),
            (node['row'] - 1, node['col'] + 1),
            (node['row'] - 1, node['col'] - 1)
        ]
        for neighbor in neighbors:
            if self._check_structure_index_validity(
                neighbor[0],
                neighbor[1]
            ):
                if not self._maze_graph.nodes[
                    self._structure_to_graph_index(
                        neighbor[0],
                        neighbor[1]
                    )
                ]['struct'].is_wall_or_chasm():
                    self._maze_graph.add_edge(
                        node['index'],
                        self._maze_graph.nodes[self._structure_to_graph_index(
                            neighbor[0],
                            neighbor[1]
                        )]['index']
                    )

    def _create_maze_graph(self):
        num_row = len(self._maze_structure)
        num_col = len(self._maze_structure[0])
        num_vertices = num_row * num_col
        self._maze_graph = nx.DiGraph()
        self._maze_graph.add_nodes_from(np.arange(
            0, num_vertices
        ))
        for i in range(num_row):
            for j in range(num_col):
                self._maze_graph.nodes[
                    self._structure_to_graph_index(i, j)
                ]['struct'] = self._maze_structure[i][j]
                self._maze_graph.nodes[
                    self._structure_to_graph_index(i, j)
                ]['row'] = i
                self._maze_graph.nodes[
                    self._structure_to_graph_index(i, j)
                ]['col'] = j
                self._maze_graph.nodes[
                    self._structure_to_graph_index(i, j)
                ]['index'] = self._structure_to_graph_index(i, j)

        for i in range(num_row):
            for j in range(num_col):
                self._add_edges_to_maze_graph(self._maze_graph.nodes[
                    self._structure_to_graph_index(i, j)
                ])

    def get_ori(self) -> float:
        return self.wrapped_env.get_ori()

    def _set_action_space(self):
        low = self.wrapped_env.action_space.low[1:]
        high = self.wrapped_env.action_space.high[1:]
        self._action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _set_observation_space(self, observation):
        spaces = {
            'scale_1' : gym.spaces.Box(
                low = np.zeros_like(observation['scale_1'], dtype = np.uint8),
                high = 255 * np.ones_like(observation['scale_1'], dtype = np.uint8),
                shape = observation['scale_1'].shape,
                dtype = observation['scale_1'].dtype
            ),   
            'scale_2' : gym.spaces.Box(
                low = np.zeros_like(observation['scale_2'], dtype = np.uint8),
                high = 255 * np.ones_like(observation['scale_2'], dtype = np.uint8),
                shape = observation['scale_2'].shape,
                dtype = observation['scale_2'].dtype
            ),   
            'sensors' : gym.spaces.Box(
                low = -np.ones_like(observation['sensors']),
                high = np.ones_like(observation['sensors']),
                shape = observation['sensors'].shape,
                dtype = observation['sensors'].dtype
            ),   
            'sampled_action' : copy.deepcopy(self._action_space),
            'inframe' : gym.spaces.Box(
                low = np.zeros_like(observation['inframe']),
                high = np.ones_like(observation['inframe']),
                shape = observation['inframe'].shape,
                dtype = observation['inframe'].dtype
            ),
            'depth' : gym.spaces.Box(
                low = np.zeros_like(observation['depth'], dtype = np.float32),
                high = np.ones_like(observation['depth'], dtype = np.float32),
                shape = observation['depth'].shape,
                dtype = observation['depth'].dtype
            ),
            'positions' : gym.spaces.Box(
                low = -np.ones_like(observation['positions']) * 40,
                high = np.ones_like(observation['positions']) * 40,
                shape = observation['positions'].shape,
                dtype = observation['positions'].dtype
            ),
            'ego_map' : gym.spaces.Box(
                low = np.zeros_like(observation['ego_map']),
                high = 255 * np.ones_like(observation['ego_map']),
                dtype = observation['ego_map'].dtype,
                shape = observation['ego_map'].shape
            )
        }
    
        if params['add_ref_scales']:
            spaces['ref_scale_1'] = gym.spaces.Box(
                low = np.zeros_like(observation['ref_scale_1'], dtype = np.uint8),
                high = 255 * np.ones_like(observation['ref_scale_1'], dtype = np.uint8),
                shape = observation['ref_scale_1'].shape,
                dtype = observation['ref_scale_1'].dtype
            )
            spaces['ref_scale_2'] = gym.spaces.Box(
                low = np.zeros_like(observation['ref_scale_2'], dtype = np.uint8),
                high = 255 * np.ones_like(observation['ref_scale_2'], dtype = np.uint8),
                shape = observation['ref_scale_2'].shape,
                dtype = observation['ref_scale_2'].dtype
            )

        self.observation_space = gym.spaces.Dict(spaces)

        return self.observation_space

    def _xy_limits(self) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = 100, 100, -100, -100
        structure = self._maze_structure
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_block():
                continue
            xmin, xmax = min(xmin, j), max(xmax, j)
            ymin, ymax = min(ymin, i), max(ymax, i)
        x0, y0 = self._init_torso_x, self._init_torso_y
        scaling = self._maze_size_scaling
        xmin, xmax = (xmin - 0.5) * scaling - x0, (xmax + 0.5) * scaling - x0
        ymin, ymax = (ymin - 0.5) * scaling - y0, (ymax + 0.5) * scaling - y0
        return xmin, xmax, ymin, ymax

    def check_angle(self, angle):
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _get_scale_indices(self, x, y, w, h, scale, size):
        center_x, center_y = x + w // 2, y + h // 2 
        x_min = center_x - size // (scale * 2) 
        x_max = center_x + size // (scale * 2) 
        y_min = center_y - size // (scale * 2) 
        y_max = center_y + size // (scale * 2) 
        if x_min < 0: 
            center_x += np.abs(x_min)
            x_max += np.abs(x_min)
            x_min = 0
        if x_max > size:
            offset = x_max - size
            center_x -= offset
            x_min -= offset
        if y_min < 0: 
            center_y += np.abs(y_min)
            y_max += np.abs(y_min)
            y_min = 0
        if y_max > size:
            offset = y_max - size
            center_y -= offset
            y_min -= offset
        return x_min, x_max, y_min, y_max

    def get_attention_window(self, frame, bbx):
        size = frame.shape[0]
        #assert frame.shape[0] == frame.shape[1]
        #window = frame.copy()
        x, y, w, h = None, None, None, None

        if len(bbx) > 0:
            x, y, w, h = bbx
        else:
            # Need to keep eye sight in the upper part of the image
            x = size // 2 - size // 4
            y = size // 2 - size // 4
            w = size // 2
            h = size // 2
        
        #print(x + w // 2, y + h // 2)
        
        # attention window computation
        scale = 3
        x_min, x_max, y_min, y_max = self._get_scale_indices(
            x, y, w, h, scale, size
        )
        window = frame[y_min:y_max, x_min:x_max].copy()
        return window

    def detect_target(self, frame):
        
        """
            Refer to the following link for reference to openCV code:
            https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)    
        target = self._task.goals[self._task.goal_index]
        mask = cv2.inRange(hsv, target.min_range , target.max_range)
        if params['debug']:
            cv2.imshow('mask', mask)
        contours, _ =  cv2.findContours(mask.copy(),
                           cv2.RETR_TREE,
                           cv2.CHAIN_APPROX_SIMPLE)
        bbx = []
        if len(contours):
            red_area = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(red_area)
            cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 1)
            bbx.extend([x, y, w, h]) 
        return bbx 

    def _get_depth(self, z_buffer):
        z_buffer = z_buffer
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        return near / (1 - z_buffer * (1 - near / far))

    def _get_point_cloud(self, depth):
        depth_img = self._get_depth(depth).copy()
        depth_img = depth_img.reshape(-1)
        points = np.zeros(depth.shape + (3,), dtype = np.float64).reshape(-1, 3)
        points[:, 1] = (self.indices_x - self.cam_mat[0, 2]) * depth_img / self.cam_mat[0, 0]
        points[:, 0] = (self.indices_y - self.cam_mat[1, 2]) * depth_img / self.cam_mat[1, 1]
        points[:, 2] = depth_img
        """
            https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py
            implement recreate scene method for faster numpy implementation of the tranformations
        """
        points = np.c_[points, np.ones(len(points))]
        points = np.dot(self.ext, points.T).T
        points = points[:, :3]
        """
        self.vec.clear()
        self.vec.extend(points)
        self.pcd.points = self.vec
        transformed_cloud = self.pcd
        transformed_cloud = transformed_cloud.crop(self.pc_target_bounds)
        #transformed_cloud = self.pcd.transform(self.c2w)
        #transformed_cloud = transformed_cloud.crop(self.pc_target_bounds)
        #transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
        #transformed_cloud.orient_normals_towards_camera_location(self.cam_pos)
        """
        return points

    def _get_bird_eye_view(self, depth):
        """
            Use the above method to create point instead of using o3d
            https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py
        """
        cloud = self._get_point_cloud(depth=depth)
        resolution = 0.1
        image = point_cloud_2_birdseye(cloud, res = resolution, side_range = (-15, 15), fwd_range=(-15, 15), height_range=(-0.4, 1))
        image = np.expand_dims(image[-15 * 15:-15 * 10, 15 * 15 // 2: 15 * 25 // 2], -1)
        """
        image = image[
            int(-30 * (10 + 5/4) / (10 * resolution)):int(-30 * (10 - 5/4) / (10 * resolution)),
            int(30 * (10 - 5/4) / (10 * resolution)): int(30 * (10 + 5/4) / (10 * resolution))
        ]
        """
        return image

    def _get_borders(self, depth, image):
        lower = np.array([0,0,0], dtype = "uint8")
        upper = np.array([180,255,40], dtype = "uint8") 
        #blur = cv2.GaussianBlur(image, (5,5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        image = cv2.bitwise_and(image, image, mask = mask)
        masked_depth = cv2.bitwise_and(depth, depth, mask = mask)
        masked_depth[masked_depth == 0] = 1
        border_view = self._get_bird_eye_view(masked_depth)
        border_view[border_view < 80] = 0 
        return masked_depth, border_view

    def _get_floor(self, depth, image):
        lower = np.array([0,0,40], dtype = "uint8")
        upper = np.array([180,255,60], dtype = "uint8") 
        #blur = cv2.GaussianBlur(image, (5,5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        image = cv2.bitwise_and(image, image, mask = mask)
        masked_depth = cv2.bitwise_and(depth, depth, mask = mask)
        masked_depth[masked_depth == 0] = 1
        floor_view = self._get_bird_eye_view(masked_depth)
        floor_view[floor_view > 80] = 0 
        return masked_depth, floor_view
    
    def _get_objects(self, depth, image):
        lower = np.array([0,50,60], dtype = "uint8")
        upper = np.array([180,255,255], dtype = "uint8") 
        #blur = cv2.GaussianBlur(image, (5,5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        image = cv2.bitwise_and(image, image, mask = mask)
        masked_depth = cv2.bitwise_and(depth, depth, mask = mask)
        masked_depth[masked_depth == 0] = 1
        object_view = self._get_bird_eye_view(masked_depth)
        object_view[object_view < 50] = 0 
        return masked_depth, object_view

    def _get_obs(self) -> np.ndarray:
        #print(self.sim.model.cam_mat0[list(self.model.camera_names).index('mtdcam1')])
        obs = self.wrapped_env._get_obs()
        #obs['front'] = cv2.resize(obs['front'], (320, 320))
        img = obs['front'].copy()
        #assert img.shape[0] == img.shape[1]
        # Target Detection and Attention Window
        bbx = self.detect_target(img)
        window = self.get_attention_window(obs['front'], bbx)
        inframe = True if len(bbx) > 0 else False 

        # Sampled Action
        sampled_action = self.get_action().astype(np.float32)

        # Sensor Readings
        ## Goal
        goal = self._task.goals[self._task.goal_index].pos - self.wrapped_env.get_xy()
        goal = np.array([
            np.linalg.norm(goal) / np.linalg.norm(self._task.goals[self._task.goal_index].pos - self._init_pos),
            self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori()) / np.pi
        ], dtype = np.float32)
        ## Velocity
        max_vel = np.array([
            self.wrapped_env.VELOCITY_LIMITS,
            self.wrapped_env.VELOCITY_LIMITS,
            self.action_space.high[0]
        ])
        sensors = np.concatenate([
            self.data.qvel.copy() / max_vel,
            np.array([self.get_ori() / np.pi, self.t / self.max_episode_size], dtype = np.float32),
            goal.copy()
        ] +  [
            (action.copy() - self.action_space.low) / (self.action_space.high - self.action_space.low) for action in self.actions
        ], -1)

        bird_eye_view = self._get_bird_eye_view(obs['front_depth'])
        if params['debug']:
            size = img.shape[0]
            cv2.imshow('stream camera', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow('stream attention window', cv2.cvtColor(cv2.resize(
                window, (size, size)
            ), cv2.COLOR_RGB2BGR))
            cv2.imshow('depth stream', (obs['front_depth'] - 0.86) / 0.14)
            #top = self.render('rgb_array')
            #cv2.imshow('position stream', top)
            cv2.imshow('bird eye view', cv2.resize(bird_eye_view, (image_width, image_height)))
            masked_depth, masked_img = self._get_borders(obs['front_depth'], obs['front'])
            cv2.imshow('masked depth borders', masked_depth)
            cv2.imshow('borders', cv2.resize(masked_img, (image_width, image_height)))
            masked_depth, masked_img = self._get_floor(obs['front_depth'], obs['front'])
            cv2.imshow('masked depth floor', masked_depth)
            cv2.imshow('floor', cv2.resize(masked_img, (image_width, image_height)))
            masked_depth, masked_img = self._get_objects(obs['front_depth'], obs['front'])
            cv2.imshow('masked depth objects', masked_depth)
            cv2.imshow('objects', cv2.resize(masked_img, (image_width, image_height)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass

        shape = window.shape[:2]
        scale_2 = cv2.resize(obs['front'], shape)
        depth = np.expand_dims(
            cv2.resize(
                obs['front_depth'].copy(), shape
            ), 0
        )

        positions = np.concatenate(self.positions + [self._task.goals[self._task.goal_index].pos], -1)

        _obs = {
            'scale_1' : window.copy(),
            'scale_2' : scale_2.copy(),
            'sensors' : sensors,
            'sampled_action' : sampled_action.copy(),
            'depth' : depth,
            'inframe' : np.array([inframe], dtype = np.float32),
            'positions' : positions.copy(),
            'ego_map' : bird_eye_view.copy()
        }

        if params['add_ref_scales']:
            ref_scale_1, ref_scale_2 = self.get_scales(obs['front'].copy(), []) 
            ref_scale_2 = cv2.resize(ref_scale_2, shape)
            _obs['ref_scale_1'] = ref_scale_1.copy()
            _obs['ref_scale_2'] = ref_scale_2.copy()

        return _obs

    def reset(self) -> np.ndarray:
        self.collision_count = 0
        self.t = 0
        self.close()
        self.set_env()
        action = self.actions[0]
        self.actions = [np.zeros_like(action) for i in range(self.n_steps)]
        goal = self._task.goals[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals = [goal.copy() for i in range(self.n_steps)]
        self.positions = [np.zeros_like(self.data.qpos) for _ in range(self.n_steps)]
        obs = self._get_obs()
        return obs

    @property
    def viewer(self) -> Any:
        if self._websock_port is not None:
            return self._mj_viewer
        else:
            return self.wrapped_env.viewer

    def _render_image(self) -> np.ndarray:
        self._mj_offscreen_viewer._set_mujoco_buffers()
        self._mj_offscreen_viewer.render(640, 480)
        return np.asarray(
            self._mj_offscreen_viewer.read_pixels(640, 480, depth=False)[::-1, :, :],
            dtype=np.uint8,
        )

    def _render_image(self) -> np.ndarray:
        self._mj_offscreen_viewer._set_mujoco_buffers()
        self._mj_offscreen_viewer.render(*self._image_shape)
        pixels = self._mj_offscreen_viewer.read_pixels(*self._image_shape, depth=False)
        return np.asarray(pixels[::-1, :, :], dtype=np.uint8)

    def _maybe_move_camera(self, viewer: Any) -> None:
        from mujoco_py import const

        if self._camera_move_x is not None:
            viewer.move_camera(const.MOUSE_ROTATE_V, self._camera_move_x, 0.0)
        if self._camera_move_y is not None:
            viewer.move_camera(const.MOUSE_ROTATE_H, 0.0, self._camera_move_y)
        if self._camera_zoom is not None:
            viewer.move_camera(const.MOUSE_ZOOM, 0, self._camera_zoom)
 
    def _find_robot(self) -> Tuple[float, float]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def _find_all_robots(self) -> List[Tuple[float, float]]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        coords = []
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                coords.append((j * size_scaling, i * size_scaling))
        return coords

    def _objball_positions(self):
        return [
            self.wrapped_env.get_body_com(name)[:2].copy() for name in self.object_balls
        ]

    def _is_in_collision(self):
        for contact in self.data.contact:
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 != 0 and geom2 != 0:
                if geom1 in self.obstacles_ids:
                    if geom2 in self.agent_ids:
                        return True
                if geom2 in self.obstacles_ids:
                    if geom1 in self.agent_ids:
                        return True
            else:
                return False

    def check_position(self, pos):
        (row, row_frac), (col, col_frac) = self._xy_to_rowcol_v2(pos[0], pos[1])
        blind = [False, False, False, False]
        collision = False
        outbound = False
        neighbors = [
            (row, col + 1),
            (row, col - 1),
            (row + 1, col),
            (row - 1, col),
        ]
        row_frac -= 0.5
        col_frac -= 0.5
        rpos = np.array([row_frac, col_frac], dtype = np.float32)
        if row > 0 and col > 0 and row < len(self._maze_structure) - 1 and col < len(self._maze_structure[0]) - 1:
            for i, (nrow, ncol) in enumerate(neighbors):
                if not self._maze_structure[nrow][ncol].is_empty():
                    rdir = (nrow - row)
                    cdir = (ncol - col)
                    direction = np.array([rdir, cdir], dtype = np.float32)
                    distance = np.dot(rpos, direction)
                    if distance > 0.325:
                        collision = True
                    if distance > 0.35:
                        blind[i] = True
        else:
            outbound = True
        #print('almost collision: {}, blind: {}'.format(collision, blind))
        return collision, blind, outbound

    def conditional_blind(self, obs, yaw, b):
        penalty = 0.0
        for direction in b:
            if direction:
                penalty += -1.0 * self._inner_reward_scaling
        
        if self._is_in_collision():
            #print('collision')
            penalty += -50.0 * self._inner_reward_scaling
            self.collision_count += 1
            obs['scale_1'] = np.zeros_like(obs['scale_1'])
            obs['scale_2'] = np.zeros_like(obs['scale_2'])
            obs['depth'] = np.zeros_like(obs['depth'])
        
        return obs, penalty

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Proprocessing and Environment Update
        action = np.clip(action, a_min = self.action_space.low, a_max = self.action_space.high)
        action = np.concatenate([
            np.array([self.target_speed], dtype = np.float32),
            action
        ], 0)
        self.t += 1
        self.total_steps += 1
        info = {}
        self.actions.pop(0)
        self.actions.append(action.copy()[1:])
        _, inner_reward, _, info = self.wrapped_env.step(action)

        # Observation and Parameter Gathering
        x, y = self.wrapped_env.get_xy()
        yaw = self.get_ori()
        v = np.linalg.norm(self.data.qvel[:2])
        self.state.set(x, y, v, yaw)
        next_pos = self.wrapped_env.get_xy()
        collision_penalty = 0.0
        next_obs = self._get_obs()

        # Computing the reward in "https://ieeexplore.ieee.org/document/8398461"
        goal = self._task.goals[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals.pop(0)
        self.goals.append(goal)
        self.positions.pop(0)
        self.positions.append(self.data.qpos.copy())
        theta_t = self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori())
        qvel = self.wrapped_env.data.qvel.copy()
        vyaw = qvel[self.wrapped_env.ORI_IND]
        vmax = self.target_speed
        if bool(next_obs['inframe'][0]):
            inner_reward = (v / vmax) * np.cos(theta_t) * (1 - (np.abs(vyaw) / params['max_vyaw']))
            inner_reward = self._inner_reward_scaling * inner_reward

        # Task Reward Computation
        outer_reward = 0
        outer_reward = self._task.reward(next_pos, bool(next_obs['inframe'][0]), self._start_pos)
        done = self._task.termination(self.wrapped_env.get_xy(),  bool(next_obs['inframe'][0]))
        info["position"] = self.wrapped_env.get_xy()

        # Collision Penalty Computation
        index = self._get_current_cell()
        self._current_cell = index
        almost_collision, blind, outbound = self.check_position(next_pos)
        if almost_collision:
            collision_penalty += -1.0 * self._inner_reward_scaling
        next_obs, penalty = self.conditional_blind(next_obs, yaw, blind)
        collision_penalty += penalty

        # Reward and Info Declaration
        if done:
            outer_reward += 200.0
            info['is_success'] = True
        else:
            info['is_success'] = False
        if outbound:
            collision_penalty += -10.0 * self._inner_reward_scaling
            next_obs['scale_1'] = np.zeros_like(obs['scale_1'])
            next_obs['scale_2'] = np.zeros_like(obs['scale_2'])
            done = True
        if self.t > self.max_episode_size:
            done = True
        
        if collision_penalty < -10.0:
            done = True

        reward = inner_reward + outer_reward + collision_penalty
        info['inner_reward'] = inner_reward
        info['outer_reward'] = outer_reward
        info['collision_penalty'] = collision_penalty
        #print('step {} reward: {}'.format(self.t, reward))
        return next_obs, reward, done, info

    def _get_current_cell(self):
        robot_x, robot_y = self.wrapped_env.get_xy()
        row, col = self._xy_to_rowcol(robot_x, robot_y)
        index = self._structure_to_graph_index(row, col)
        return index

    def close(self) -> None:
        self.wrapped_env.close()
        if self._websock_server_pipe is not None:
            self._websock_server_pipe.send(None)

    def xy_to_imgrowcol(self, x, y): 
        (row, row_frac), (col, col_frac) = self._xy_to_rowcol_v2(x, y)
        row = self.top_view_size * row + int(row_frac * self.top_view_size)
        col = self.top_view_size * col + int(col_frac * self.top_view_size)
        return row, col 

    def render(self, mode = 'human', **kwargs):
        if mode == 'rgb_array':
            return self.get_top_view()
        elif mode == "human" and self._websock_port is not None:
            if self._mj_offscreen_viewer is None:
                from mujoco_py import MjRenderContextOffscreen as MjRCO
                from mujoco_maze.websock_viewer import start_server

                self._mj_offscreen_viewer = MjRCO(self.wrapped_env.sim)
                self._maybe_move_camera(self._mj_offscreen_viewer)
                self._websock_server_pipe = start_server(self._websock_port)
            return self._websock_server_pipe.send(self._render_image())
        else:
            if self.wrapped_env.viewer is None:
                self.wrapped_env.render(mode, **kwargs)
                self._maybe_move_camera(self.wrapped_env.viewer)
            return self.wrapped_env.render(mode, **kwargs) 

    def get_top_view(self):
        block_size = self.top_view_size

        img = np.zeros(
            (int(block_size * len(self._maze_structure)), int(block_size * len(self._maze_structure[0])), 3),
            dtype = np.uint8
        )

        for i in range(len(self._maze_structure)):
            for j in range(len(self._maze_structure[0])):
                if  self._maze_structure[i][j].is_wall_or_chasm():
                    img[
                        int(block_size * i): int(block_size * (i + 1)),
                        int(block_size * j): int(block_size * (j + 1))
                    ] = 128


        def xy_to_imgrowcol(x, y):
            (row, row_frac), (col, col_frac) = self._xy_to_rowcol_v2(x, y)
            row = block_size * row + int((row_frac) * block_size)
            col = block_size * col + int((col_frac) * block_size)
            return int(row), int(col)

        pos = self.wrapped_env.get_xy() 
        ori = self.wrapped_env.get_ori()
        if ori < 0 and ori > -np.pi:
            ori += 2 * np.pi
        row, col = xy_to_imgrowcol(pos[0], pos[1])
        
        pt1_x = pos[0] + self._maze_size_scaling * 0.3 * np.cos(ori)
        pt1_y = pos[1] + self._maze_size_scaling * 0.3 * np.sin(ori)
        pt2_x = pos[0] + self._maze_size_scaling * 0.15 * np.cos(2 * np.pi / 3 + ori)
        pt2_y = pos[1] + self._maze_size_scaling * 0.15 * np.sin(2 * np.pi / 3 + ori)
        pt3_x = pos[0] + self._maze_size_scaling * 0.15 * np.cos(4 * np.pi / 3 + ori)
        pt3_y = pos[1] + self._maze_size_scaling * 0.15 * np.sin(4 * np.pi / 3 + ori)

        pt1 = xy_to_imgrowcol(pt1_y, pt1_x)
        pt2 = xy_to_imgrowcol(pt2_y, pt2_x)
        pt3 = xy_to_imgrowcol(pt3_y, pt3_x)

        triangle_cnt = np.array( [pt1, pt2, pt3] )
        cv2.drawContours(img, [triangle_cnt], 0, (255,255,255), -1)

        #img[row - int(block_size / 10): row + int(block_size / 20), col - int(block_size / 20): col + int(block_size / 20)] = [255, 255, 255]
        for i, goal in enumerate(self._task.goals):
            pos = goal.pos
            row, col = xy_to_imgrowcol(pos[0], pos[1])
            if i == self._task.goal_index:
                img[
                    row - int(block_size / 10): row + int(block_size / 10),
                    col - int(block_size / 10): col + int(block_size / 10)
                ] = [0, 0, 255]
            else:
                img[
                    row - int(block_size / 10): row + int(block_size / 10),
                    col - int(block_size / 10): col + int(block_size / 10)
                ] = [0, 255, 0]

        return np.flipud(img)


class DiscreteMazeEnv(MazeEnv):
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
        max_episode_size: int = 2000,
        n_steps = 50,
        include_position: bool = True,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        inner_reward_scaling: float = 1.0,
        restitution_coef: float = 0.8,
        task_kwargs: dict = {},
        websock_port: Optional[int] = None,
        camera_move_x: Optional[float] = None,
        camera_move_y: Optional[float] = None,
        camera_zoom: Optional[float] = None,
        image_shape: Tuple[int, int] = (600, 480),
        **kwargs,
    ) -> None:
        super(DiscreteMazeEnv, self).__init__(
            model_cls=model_cls,
            maze_task=maze_task,
            max_episode_size=max_episode_size,
            n_steps=n_steps,
            include_position=include_position,
            maze_height=maze_height,
            maze_size_scaling=maze_size_scaling,
            inner_reward_scaling=inner_reward_scaling,
            restitution_coef=restitution_coef,
            task_kwargs=task_kwargs,
            websock_port=websock_port,
            camera_move_x=camera_move_x,
            camera_move_y=camera_move_y,
            camera_zoom=camera_zoom,
            image_shape=image_shape,
            **kwargs,
        )

    def _set_action_space(self):
        self._action_space = gym.spaces.MultiDiscrete([2, 13])
 
    def discrete_vyaw(self, vyaw):
        if vyaw < -0.125:
            vyaw = 0
        elif vyaw < 0.125:
            vyaw = 1
        else:
            vyaw = 2
        return vyaw

    def get_action(self):
        ai = proportional_control(self.target_speed, self.state.v)
        di, self.target_ind = pure_pursuit_steer_control(
            self.state, self.target_course, self.target_ind
        )
        #yaw = self.state.yaw +  self.state.v / self.state.WB * math.tan(di) * self.dt
        v = self.state.v + ai * self.dt
        vyaw = self.state.v / self.state.WB * math.tan(di)
        #self.state.update(ai, di, self.dt)
        #v = self.state.v
        #yaw = self.state.yaw
        # Refer to simulations/point PointEnv: def step() for more information
        vyaw = self.discrete_vyaw(vyaw)
        self.sampled_action = np.array([
            1,
            vyaw,
        ], dtype = np.float32)
        return self.sampled_action

    def continuous_action(self, action):
        move, omega = action
        if move == 0:
            move = 0
        else:
            move = self.target_speed
        if omega == 0:
            omega = -1.5
        elif omega == 1:
            omega = 0.0
        elif omega == 2:
            omega = 1.5
        else:
            raise ValueError

        action = np.array([move, omega], dtype = np.float32)
        return action

    def _get_obs(self) -> np.ndarray:
        #print(self.sim.model.cam_mat0[list(self.model.camera_names).index('mtdcam1')])
        obs = self.wrapped_env._get_obs()
        #obs['front'] = cv2.resize(obs['front'], (320, 320))
        img = obs['front'].copy()
        #assert img.shape[0] == img.shape[1]
        # Target Detection and Attention Window
        bbx = self.detect_target(img)
        window = self.get_attention_window(obs['front'], bbx)
        inframe = True if len(bbx) > 0 else False 

        # Sampled Action
        sampled_action = self.get_action().astype(np.float32)

        # Sensor Readings
        ## Goal
        goal = self._task.goals[self._task.goal_index].pos - self.wrapped_env.get_xy()
        goal = np.array([
            np.linalg.norm(goal) / np.linalg.norm(self._task.goals[self._task.goal_index].pos - self._init_pos),
            self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori()) / np.pi
        ], dtype = np.float32)
        ## Velocity
        max_vel = np.array([
            self.wrapped_env.VELOCITY_LIMITS,
            self.wrapped_env.VELOCITY_LIMITS,
            self.wrapped_env.action_space.high[1]
        ])
        low = self.wrapped_env.action_space.low[0]
        high = self.wrapped_env.action_space.high[0]
        sensors = np.concatenate([
            self.data.qvel.copy() / max_vel,
            np.array([self.get_ori() / np.pi, self.t / self.max_episode_size], dtype = np.float32),
            goal.copy()
        ] +  [
            (action.copy() - low) / (high - low) for action in self.actions
        ], -1)

        bird_eye_view = self._get_bird_eye_view(obs['front_depth'])
        if params['debug']:
            size = img.shape[0]
            cv2.imshow('stream camera', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow('stream attention window', cv2.cvtColor(cv2.resize(
                window, (size, size)
            ), cv2.COLOR_RGB2BGR))
            cv2.imshow('depth stream', (obs['front_depth'] - 0.86) / 0.14)
            top = self.render('rgb_array')
            cv2.imshow('position stream', top)
            cv2.imshow('bird eye view', cv2.resize(bird_eye_view, (image_width, image_height)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass

        shape = window.shape[:2]
        scale_2 = cv2.resize(obs['front'], shape)
        depth = np.expand_dims(
            cv2.resize(
                obs['front_depth'].copy(), shape
            ), 0
        )

        positions = np.concatenate(self.positions + [self._task.goals[self._task.goal_index].pos], -1)
        
        _obs = {
            'scale_1' : window.copy(),
            'scale_2' : scale_2.copy(),
            'sensors' : sensors,
            'sampled_action' : sampled_action.copy(),
            'depth' : depth,
            'inframe' : np.array([inframe], dtype = np.float32),
            'positions' : positions.copy(),
            'ego_map' : bird_eye_view.copy()
        }

        if params['add_ref_scales']:
            ref_scale_1, ref_scale_2 = self.get_scales(obs['front'].copy(), []) 
            ref_scale_2 = cv2.resize(ref_scale_2, shape)
            _obs['ref_scale_1'] = ref_scale_1.copy()
            _obs['ref_scale_2'] = ref_scale_2.copy()

        return _obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Proprocessing and Environment Update
        assert self.action_space.contains(action)
        action = self.continuous_action(action)
        self.t += 1
        self.total_steps += 1
        info = {}
        self.actions.pop(0)
        self.actions.append(action.copy()[1:])
        _, inner_reward, _, info = self.wrapped_env.step(action)

        # Observation and Parameter Gathering
        x, y = self.wrapped_env.get_xy()
        yaw = self.get_ori()
        v = np.linalg.norm(self.data.qvel[:2])
        self.state.set(x, y, v, yaw)
        next_pos = self.wrapped_env.get_xy()
        collision_penalty = 0.0
        next_obs = self._get_obs()

        # Computing the reward in "https://ieeexplore.ieee.org/document/8398461"
        goal = self._task.goals[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals.pop(0)
        self.goals.append(goal)
        self.positions.pop(0)
        self.positions.append(self.data.qpos.copy())
        theta_t = self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori())
        qvel = self.wrapped_env.data.qvel.copy()
        vyaw = qvel[self.wrapped_env.ORI_IND]
        vmax = self.target_speed
        if bool(next_obs['inframe'][0]):
            inner_reward = (v / vmax) * np.cos(theta_t) * (1 - (np.abs(vyaw) / params['max_vyaw']))
            inner_reward = self._inner_reward_scaling * inner_reward

        # Task Reward Computation
        outer_reward = 0
        outer_reward = self._task.reward(next_pos, bool(next_obs['inframe'][0]), self._start_pos)
        done = self._task.termination(self.wrapped_env.get_xy(),  bool(next_obs['inframe'][0]))
        info["position"] = self.wrapped_env.get_xy()

        # Collision Penalty Computation
        index = self._get_current_cell()
        self._current_cell = index
        almost_collision, blind, outbound = self.check_position(next_pos)
        if almost_collision:
            collision_penalty += -1.0 * self._inner_reward_scaling
        next_obs, penalty = self.conditional_blind(next_obs, yaw, blind)
        collision_penalty += penalty

        # Reward and Info Declaration
        if done:
            outer_reward += 200.0
            info['is_success'] = True
        else:
            info['is_success'] = False
        if outbound:
            collision_penalty += -10.0 * self._inner_reward_scaling
            next_obs['scale_1'] = np.zeros_like(obs['scale_1'])
            next_obs['scale_2'] = np.zeros_like(obs['scale_2'])
            done = True
        if self.t > self.max_episode_size:
            done = True
        
        if collision_penalty < -10.0:
            done = True

        reward = inner_reward + outer_reward + collision_penalty
        info['inner_reward'] = inner_reward
        info['outer_reward'] = outer_reward
        info['collision_penalty'] = collision_penalty
        #print('step {} reward: {}'.format(self.t, reward))
        return next_obs, reward, done, info
    

def _add_object_ball(
    worldbody: ET.Element, i: str, j: str, x: float, y: float, size: float
) -> None:
    body = ET.SubElement(worldbody, "body", name=f"objball_{i}_{j}", pos=f"{x} {y} 0")
    mass = 0.0001 * (size ** 3)
    ET.SubElement(
        body,
        "geom",
        type="sphere",
        name=f"objball_{i}_{j}_geom",
        size=f"{size}",  # Radius
        pos=f"0.0 0.0 {size}",  # Z = size so that this ball can move!!
        rgba=maze_task.BLUE.rgba_str(),
        contype="1",
        conaffinity="1",
        solimp="0.9 0.99 0.001",
        mass=f"{mass}",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"objball_{i}_{j}_x",
        axis="1 0 0",
        pos="0 0 0.0",
        type="slide",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"objball_{i}_{j}_y",
        axis="0 1 0",
        pos="0 0 0",
        type="slide",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"objball_{i}_{j}_rot",
        axis="0 0 1",
        pos="0 0 0",
        type="hinge",
        limited="false",
    )

def _add_movable_block(
    worldbody: ET.Element,
    struct: maze_env_utils.MazeCell,
    i: str,
    j: str,
    size_scaling: float,
    x: float,
    y: float,
    h: float,
    height_offset: float,
) -> None:
    falling = struct.can_move_z()
    if struct.can_spin():
        h *= 0.1
        x += size_scaling * 0.25
        shrink = 0.1
    elif falling:
        # The "falling" blocks are shrunk slightly and increased in mass to
        # ensure it can fall easily through a gap in the platform blocks.
        shrink = 0.99
    elif struct.is_half_block():
        shrink = 0.5
    else:
        shrink = 1.0
    size = size_scaling * 0.5 * shrink
    movable_body = ET.SubElement(
        worldbody,
        "body",
        name=f"movable_{i}_{j}",
        pos=f"{x} {y} {h}",
    )
    ET.SubElement(
        movable_body,
        "geom",
        name=f"block_{i}_{j}",
        pos="0 0 0",
        size=f"{size} {size} {h}",
        type="box",
        material="",
        mass="0.001" if falling else "0.0002",
        contype="1",
        conaffinity="1",
        rgba="0.9 0.1 0.1 1",
    )
    if struct.can_move_x():
        ET.SubElement(
            movable_body,
            "joint",
            axis="1 0 0",
            name=f"movable_x_{i}_{j}",
            armature="0",
            damping="0.0",
            limited="true" if falling else "false",
            range=f"{-size_scaling} {size_scaling}",
            margin="0.01",
            pos="0 0 0",
            type="slide",
        )
    if struct.can_move_y():
        ET.SubElement(
            movable_body,
            "joint",
            armature="0",
            axis="0 1 0",
            damping="0.0",
            limited="true" if falling else "false",
            range=f"{-size_scaling} {size_scaling}",
            margin="0.01",
            name=f"movable_y_{i}_{j}",
            pos="0 0 0",
            type="slide",
        )
    if struct.can_move_z():
        ET.SubElement(
            movable_body,
            "joint",
            armature="0",
            axis="0 0 1",
            damping="0.0",
            limited="true",
            range=f"{-height_offset} 0",
            margin="0.01",
            name=f"movable_z_{i}_{j}",
            pos="0 0 0",
            type="slide",
        )
    if struct.can_spin():
        ET.SubElement(
            movable_body,
            "joint",
            armature="0",
            axis="0 0 1",
            damping="0.0",
            limited="false",
            name=f"spinable_{i}_{j}",
            pos="0 0 0",
            type="ball",
        )

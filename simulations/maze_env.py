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
from typing import Any, List, Optional, Tuple, Type, Callable
import gym
import numpy as np
import networkx as nx
from neurorobotics.simulations import maze_env_utils, maze_task
from neurorobotics.simulations.agent_model import AgentModel
from neurorobotics.utils.env_utils import calc_spline_course, TargetCourse, proportional_control, \
    State, pure_pursuit_steer_control
import random
import copy
from neurorobotics.constants import params, image_width, image_height
import math
import cv2
import colorsys
from neurorobotics.simulations.maze_task import Rgb
import open3d as o3d
from neurorobotics.utils.point_cloud import rotMatList2NPRotMat
from neurorobotics.simulations.maze_env_utils import MazeCell

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.join(os.getcwd(), 'neurorobotics/assets', 'xml')


def scale_to_255(a, minimum, maximum, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
    
    :param mininum: mininum of range to normalize
    :type mininum: int or float or uint8
    :param maximum: maximum of range to normalize
    :type maximum: int or float or uint8
    :param dtype: data type of data (default is uint8)
    :type dtype: np.dtype
    """
    return (((a - minimum) / float(maximum - minimum)) * 255).astype(dtype)

def sort_arrays(x, y, z):
    """Sorts x and y according to index of z.
    
    :param x: first input
    :type x: np.ndarray
    :param y: second input
    :type y: np.ndarray
    :param z: array to sort
    :type z: np.ndarray
    :return: (x, y, z)
    :rtype: Tuple[Type[np.ndarray]]
    """
    indices = z.argsort()
    z = z[indices]
    x = x[indices]
    y = y[indices]
    return x, y, z

ANGLE_EPS = 0.001

def normalize(v):
    """Normalize array.
    
    :param v: 
    """
    return v / np.linalg.norm(v)

def get_r_matrix(ax_, angle):
    """ Description.
    
    :param ax_:
    :type ax_:
    :param angle:
    :type angle:
    :return:
    :rtype: np.ndarray
    """
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32)
        R = np.eye(3) + np.sin(angle) * S_hat + \
            (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
    else:
        R = np.eye(3)
    return R

def transform_pose(points, current_pose):
    """Transforms the point cloud into geocentric frame to account for camera position.
    
    :param points: (x, y, z)
    :type points: np.ndarray
    :param current_pose: camera_pose (x, y, theta, radians))
    :type current_pose: np.ndarray
    :return: transformed pose (x, y, z)
    :rtype: np.ndarray
    """
    R = get_r_matrix([0., 0., 1.], angle=current_pose[2])
    points = np.matmul(points, R.T)
    points[:, 0] = points[:, 0] + current_pose[0]
    points[:, 1] = points[:, 1] + current_pose[1]
    return points

class MazeEnv(gym.Env):
    """Base Class for a stochastic maze environment.
    
    :param model_cls: Class of agent to spawn
    :type model_cls: Type[AgentModel]
    :param maze_task_generator: generator method for sampling a random a maze task
    :type maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
    :param max_episode_size: maximum number of steps permissible in an episode
    :type max_episode_size: int = 2000,
    :param n_steps: number of steps in the past to store state for
    :type n_steps: int = 50,
    :param include_position: 
    :type include_position: bool = True,
    :param maze_height: height of maze in simulations 
    :type maze_height: float = 0.5,
    :param maze_size_scaling: 
    :type maze_size_scaling: float = 4.0,
    :param inner_reward_scaling: 
    :type inner_reward_scaling: float = 1.0,
    :param restitution_coef: 
    :type restitution_coef: float = 0.8,
    :param task_kwargs: 
    :type task_kwargs: dict = {},
    :param websock_port: 
    :type websock_port: Optional[int] = None,
    :param camera_move_x: 
    :type camera_move_x: Optional[float] = None,
    :param camera_move_y: 
    :type camera_move_y: Optional[float] = None,
    :param camera_zoom: 
    :type camera_zoom: Optional[float] = None,
    :param image_shape: 
    :type image_shape: Tuple[int, int] = (600, 480),
    :param mode: 
    :type mode: Optional[int]= None,
    """
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task_generator: Callable,
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
        mode=None,
        **kwargs,
    ) -> None:
        """INITIALIZE.
        """
        self.mode = mode
        self.collision_count = 0
        self.n_steps = n_steps
        self.kwargs = kwargs
        self.top_view_size = params['top_view_size']
        self.t = 0  # time steps
        self.total_steps = 0
        self.total_eps = 0
        self.ep = 0
        self.max_episode_size = max_episode_size
        self._maze_task_generator = maze_task_generator
        self._task_kwargs = task_kwargs
        self.elevated = any(maze_env_utils.MazeCell.CHASM in row for row in self._maze_structure)
        # Are there any movable blocks?
        self.blocks = any(any(r.can_move() for r in row) for row in self._maze_structure)
        self._maze_height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self._inner_reward_scaling = inner_reward_scaling
        
        # Observe other objectives
        self._restitution_coef = restitution_coef
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
        """
        """
        target_pos = self._task.objects[self._task.goal_index].pos
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
                if struct.is_robot():
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

        self._task, self._maze_structure, self._open_position_indices = self._maze_task_generator(self._maze_size_scaling, **self._task_kwargs)
        for i, goal in enumerate(self._task.objects):
            z = goal.pos[2] if goal.dim >= 3 else 0.1*self._maze_size_scaling
            if goal.custom_size is None:
                size = f"{self._maze_size_scaling * 0.1}"
            else:
                if isinstance(goal.custom_size, list):
                    size = ' '.join(map(str, goal.custom_size))
                else:
                    size = f"{goal.custom_size}"
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
        self.cam_mat = np.array(
                [[f, 0, cx], [0, f, cy], [0, 0, 1]],
                dtype=np.float32) 
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
        self.cam_pos = self.model.body_pos[self.cam_body_id] + np.array([0, 0, 0.5])
        mat = rotMatList2NPRotMat(self.sim.model.cam_mat0[index])
        rot_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        mat = np.dot(mat, rot_mat)
        ext = np.eye(4)
        ext[:3, :3] = mat
        ext[:3, 3] = self.cam_pos
        self.ext = ext
        min_bound = [-35, -35, 0.0]
        max_bound = [35, 35, 1.5]
        self.pc_target_bounds = np.array([min_bound, max_bound], dtype=np.float32)

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
        goal = self._task.objects[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals = [goal.copy() for _ in range(self.n_steps)]
        self.positions = [np.zeros_like(self.data.qpos) for _ in range(self.n_steps)]
        self._create_maze_graph()
        self.sampled_path = self._sample_path()
        self._current_cell = copy.deepcopy(self.sampled_path[0])
        self._find_all_waypoints()
        self._find_cubic_spline_path()
        self._setup_vel_control()
        self.resolution = 0.5
        self.ego_map_side_range = (-15, 15)
        self.ego_map_fwd_range = (0, 30)
        self.height_range = (0, 1.5)
        self.allo_map_side_range = side_range = (-40, 40)
        self.allo_map_fwd_range = fwd_range = (-40, 40)
        self.allo_map_height_range = (0, 1.5)
        x_max = 1 + int((side_range[1] - side_range[0]) / self.resolution)
        y_max = 1 + int((fwd_range[1] - fwd_range[0]) / self.resolution)
        self.map = np.zeros([y_max, x_max, 3], dtype=np.uint8)
        self.maps = [self.map.copy()] * self.n_steps
        self.reward = 0.0
        self.loc_map = [self.get_local_map(self.map).copy()] * self.n_steps
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
        self.wx.append(self._task.objects[self._task.goal_index].pos[0])
        self.wy.append(self._task.objects[self._task.goal_index].pos[1])
        self.final = [self.wx[-1], self.wy[-1]]

    def _find_cubic_spline_path(self):
        self.cx, self.cy, self.cyaw, self.ck, self.s = calc_spline_course(self.wx, self.wy, params['ds'])

    @property
    def action_space(self):
        return self._action_space

    def _setup_vel_control(self):
        self.target_speed = 2
        self.state = State(
            x=self.wrapped_env.sim.data.qpos[0],
            y=self.wrapped_env.sim.data.qpos[1],
            yaw=self.wrapped_env.sim.data.qpos[2],
            v=np.linalg.norm(self.wrapped_env.sim.data.qvel[:2]),
            WB=0.2 * self._maze_size_scaling,
        )
        self.last_idx = len(self.cx) - 1
        self.target_course = TargetCourse(self.cx, self.cy)
        self.target_ind, _ = self.target_course.search_target_index(self.state)

    def _sample_path(self):
        robot_x, robot_y = self.wrapped_env.get_xy()
        row, col = self._xy_to_rowcol(robot_x, robot_y)
        source = self._structure_to_graph_index(row, col)
        goal_pos = self._task.objects[self._task.goal_index].pos[:2]
        row, col = self._xy_to_rowcol(goal_pos[0], goal_pos[1])
        target = self._structure_to_graph_index(row, col)
        paths = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(
            self._maze_graph,
            source,
            target
        ))
        return paths[0]

    def get_action(self):
        di, self.target_ind = pure_pursuit_steer_control(
            self.state, self.target_course, self.target_ind
        )
        # yaw = self.state.yaw +  self.state.v / self.state.WB * math.tan(di) * self.dt
        vyaw = self.state.v / self.state.WB * math.tan(di)
        """
        self.state.update(ai, di, self.dt)
        v = self.state.v
        yaw = self.state.yaw
        # Refer to simulations/point PointEnv: def step() for more information
        yaw = self.check_angle(self.state.yaw + vyaw * self.dt)
        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)
        """
        self.sampled_action = np.array([
            vyaw,
        ], dtype=np.float32)
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
            'scaled_sampled_action' : gym.spaces.Box(
                low = -np.ones_like(observation['scaled_sampled_action']),
                high = np.ones_like(observation['scaled_sampled_action']),
                shape = observation['scaled_sampled_action'].shape,
                dtype = observation['scaled_sampled_action'].dtype
            ),
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
            'loc_map' : gym.spaces.Box(
                low = np.zeros_like(observation['loc_map']),
                high = 255 * np.ones_like(observation['loc_map']),
                dtype = observation['loc_map'].dtype,
                shape = observation['loc_map'].shape
            ),
            'prev_loc_map' : gym.spaces.Box(
                low = np.zeros_like(observation['prev_loc_map']),
                high = 255 * np.ones_like(observation['prev_loc_map']),
                dtype = observation['prev_loc_map'].dtype,
                shape = observation['prev_loc_map'].shape
            ),
            'bbx' : gym.spaces.Box(
                low = np.zeros_like(observation['bbx']),
                high = np.ones_like(observation['bbx']) * np.array([image_width, image_height, image_width, image_height], dtype = np.float32),
                shape = observation['bbx'].shape,
                dtype = observation['bbx'].dtype
            ),
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
        # assert frame.shape[0] == frame.shape[1]
        # window = frame.copy()
        x, y, w, h = None, None, None, None

        if len(bbx) > 0:
            x, y, w, h = bbx
            bbx = np.array(bbx).copy()
        else:
            # Need to keep eye sight in the upper part of the image
            x = size // 2 - size // 4
            y = size // 2 - size // 4
            w = size // 2
            h = size // 2
            bbx = np.array([x, y, w, h]).copy() 
        # attention window computation
        scale = 3
        x_min, x_max, y_min, y_max = self._get_scale_indices(
            x, y, w, h, scale, size
        )
        window = frame[y_min:y_max, x_min:x_max].copy()
        return window, bbx

    def detect_color(
            self,
            frame: np.ndarray,
            display: bool = False
            ):
        """Localize and classify color objects in scene.
        
        :param frame: Visual Perception Input
        :type frame: np.ndarray
        :param display: Switch to display frames for debugging
        :type display: bool
        :return: Bounding boxes and their classes detected in the frame
        :rtype: Tuple[List[List], List[int]]
        """
        Y, X, _ = frame.shape
        boxes = []
        info = []
        boxes = []
        results = None
        masks = {}
        rgbs = params['available_rgb'] + [params['target_rgb']]
        for i, rgb in enumerate(rgbs):
            h, s, _ = colorsys.rgb_to_hsv(*rgb)
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
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(
                    hsv_frame,
                    np.array(hsv_low, dtype=np.int32),
                    np.array(hsv_high, dtype=np.int32))
            if display:
                masks['r: {:.2f}, g: {:.2f}, b: {:.2f}'.format(
                        rgb[0],
                        rgb[1],
                        rgb[2]
                        )] = mask.copy()
                if results is None:
                    results = mask
                else:
                    results += mask
            contours, _ = cv2.findContours(
                    mask.copy(),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
            bbx = []
            if len(contours):
                color_area = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(color_area)
                x = (x + w / 2) / X
                y = (y + h / 2) / Y
                w = w / X
                h = h / Y
                if display:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.putText(
                            frame,
                            "class: {}".format(i),
                            (x + w, y + h + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 1, cv2.LINE_AA)
                bbx.extend([x, y, w, h])
                boxes.append(bbx)
                info.append(i)

        if display:
            results = np.clip(results, 0, 255).astype(np.uint8)
            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imshow('mask', results)
            """
            for key, mask in masks.items():
                cv2.imshow(key, mask)
            """
        return boxes, info

    def detect_target(self, frame):
        
        """
            Refer to the following link for reference to openCV code:
            https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        target = self._task.objects[self._task.goal_index]
        mask = cv2.inRange(hsv, target.min_range, target.max_range)
        contours, _ = cv2.findContours(
                mask.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        bbx = []
        if len(contours):
            red_area = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(red_area)
            # cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 1)
            bbx.extend([x, y, w, h])
        return bbx

    def _get_depth(self, z_buffer):
        z_buffer = z_buffer
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        return near / (1 - z_buffer * (1 - near / far))

    def _get_point_cloud(self, depth):
        depth_img = self._get_depth(depth)
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

    def process_point_cloud(self,
        points,
        side_range=(-10., 10.),
        fwd_range = (-10., 10.),
    ):
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]

        # FILTER - To return only indices of points within desired cube
        # Three filters for: Front-to-back, side-to-side, and height ranges
        # Note left side is positive y axis in LIDAR coordinates
        f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
        s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
        filter = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filter).flatten()

        # KEEPERS
        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        return x_points, y_points, z_points

    def _cloud2map(self,
        points,
        res = 0.1, 
        side_range=(-10., 10.),
        fwd_range = (-10., 10.),
        height_range=(-2., 2.),
    ):
       
        x_points, y_points, z_points = self.process_point_cloud(
            points=points,
            side_range=side_range,
            fwd_range=fwd_range,
        )
        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.ceil(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = np.clip(a=z_points,
                               a_min=height_range[0],
                               a_max=height_range[1])

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values = scale_to_255(pixel_values,
                                    mininum=height_range[0],
                                    maximum=height_range[1])

        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = 1 + int((side_range[1] - side_range[0]) / res)
        y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
        im = np.zeros([y_max, x_max], dtype=np.uint8)

        x_img, y_img, pixel_values = sort_arrays(x_img, y_img, pixel_values)
        #pixel_values[:] = 255
        im[y_img, x_img] = pixel_values

        return im
    
    def get_coverage(self, allo_map):
        explored = allo_map > 0
        coverage = np.sum(explored) / (explored.shape[0] * explored.shape[1])
        return coverage

    def _get_borders(self, cloud, hsv):
        lower = np.array([0,0,0], dtype = "uint8")
        upper = np.array([180,40,10], dtype = "uint8") 
        #blur = cv2.GaussianBlur(image, (5,5), 0)
        mask = np.repeat(
            np.expand_dims(
                cv2.inRange(hsv, lower, upper).reshape(-1), -1
            ), cloud.shape[-1], -1
        )
        cloud = cv2.bitwise_and(cloud, cloud, mask = mask)
        cloud = cloud[~np.all(cloud == 0, axis=1)]
        return cloud

    def _get_floor(self, cloud, hsv):
        lower = np.array([0,0,15], dtype = "uint8")
        upper = np.array([180, 40,60], dtype = "uint8") 
        #blur = cv2.GaussianBlur(image, (5,5), 0)
        mask = np.repeat(
            np.expand_dims(
                cv2.inRange(hsv, lower, upper).reshape(-1), -1
            ), cloud.shape[-1], -1
        )
        cloud = cv2.bitwise_and(cloud, cloud, mask = mask)
        cloud = cloud[~np.all(cloud == 0, axis=1)]
        cloud[:, 2] = 10 * (self.height_range[1] - self.height_range[0]) / 255 + self.height_range[0]
        return cloud
    
    def _get_objects(self, cloud, hsv):
        lower = np.array([0,40,60], dtype = "uint8")
        upper = np.array([180,255,255], dtype = "uint8") 
        #blur = cv2.GaussianBlur(image, (5,5), 0)
        mask = np.repeat(
            np.expand_dims(
                cv2.inRange(hsv, lower, upper).reshape(-1), -1
            ), cloud.shape[-1], -1
        )
        cloud = cv2.bitwise_and(cloud, cloud, mask = mask)
        cloud = cloud[~np.all(cloud == 0, axis=1)]
        return cloud

    def _get_target(self, cloud, hsv):
        target = self._task.objects[self._task.goal_index] 
        lower = target.min_range
        upper = target.max_range
        mask = np.repeat(
            np.expand_dims(
                cv2.inRange(hsv, lower, upper).reshape(-1), -1
            ), cloud.shape[-1], -1
        )
        cloud = cv2.bitwise_and(cloud, cloud, mask = mask)
        cloud = cloud[~np.all(cloud == 0, axis=1)]
        return cloud

    def get_ego_maps(self, depth, rgb):
        """
            Use the above method to create point instead of using o3d
            https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py
        """
        cloud = self._get_point_cloud(depth=depth)
        complete_ego_map = self._cloud2map(
            cloud,
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )
        
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        borders_cloud = self._get_borders(cloud, hsv)
        border_ego_map = self._cloud2map(borders_cloud,
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )
        
        floor_cloud = self._get_floor(cloud, hsv)
        floor_ego_map = self._cloud2map(floor_cloud,
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )
        floor_ego_map[floor_ego_map > 0] = 255 - floor_ego_map[floor_ego_map > 0]
        
        objects_cloud = self._get_objects(cloud, hsv)
        objects_ego_map = self._cloud2map(objects_cloud, 
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )
        target_cloud = self._get_target(cloud, hsv)
        target_ego_map = self._cloud2map(target_cloud,
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )
        return complete_ego_map, border_ego_map, floor_ego_map, objects_ego_map, target_ego_map

    def get_ego_clouds(self, depth, rgb):
        cloud = self._get_point_cloud(depth=depth)
        xy = self.data.qpos[:2]
        R = get_r_matrix([0., 0., 1.], angle=self.data.qpos[2])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        borders_cloud = self._get_borders(cloud, hsv)
        borders_cloud = self.transform_cloud_pose(np.stack(self.process_point_cloud(
            self._get_borders(cloud, hsv),
            side_range=self.ego_map_side_range,
            fwd_range=self.ego_map_fwd_range
        ), -1), xy, R)
        floor_cloud = self.transform_cloud_pose(np.stack(self.process_point_cloud(
            self._get_floor(cloud, hsv),
            side_range=self.ego_map_side_range,
            fwd_range=self.ego_map_fwd_range
        ), -1), xy, R)
        objects_cloud = self.transform_cloud_pose(np.stack(self.process_point_cloud(
            self._get_objects(cloud, hsv),
            side_range=self.ego_map_side_range,
            fwd_range=self.ego_map_fwd_range
        ), -1), xy, R)
        return borders_cloud, floor_cloud, objects_cloud

    def get_ego_map(self, g_borders_cloud, g_floor_cloud, g_objects_cloud):
        R = R = get_r_matrix([0., 0., 1.], angle=-self.data.qpos[2]) 
        xy = self.data.qpos[:2]
        borders_cloud = self.inv_transform_cloud_pose(g_borders_cloud, xy, R)
        floor_cloud = self.inv_transform_cloud_pose(g_floor_cloud, xy, R)
        objects_cloud = self.inv_transform_cloud_pose(g_objects_cloud, xy, R)
        
        border_ego_map = self._cloud2map(borders_cloud,
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )

        floor_ego_map = self._cloud2map(floor_cloud,
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )

        objects_ego_map = self._cloud2map(objects_cloud, 
            res = self.resolution,
            side_range = self.ego_map_side_range,
            fwd_range = self.ego_map_fwd_range,
            height_range = self.height_range
        )
        return np.stack([border_ego_map, floor_ego_map, objects_ego_map], -1)

    def inv_transform_cloud_pose(self, points, xy, R):
        """
        Transforms the point cloud into geocentric frame to account for
        camera position
        Input:
            points                  : ...x3
            current_pose            : camera position (x, y, theta (radians))
        Output:
            XYZ : ...x3
        """
        points[:, 0] = points[:, 0] - xy[0]
        points[:, 1] = points[:, 1] - xy[1]
        points = np.matmul(points, R.T)
        return points
    
    def transform_cloud_pose(self, points, xy, R):
        """
        Transforms the point cloud into geocentric frame to account for
        camera position
        Input:
            points                  : ...x3
            current_pose            : camera position (x, y, theta (radians))
        Output:
            XYZ : ...x3
        """
        points = np.matmul(points, R.T)
        points[:, 0] = points[:, 0] + xy[0]
        points[:, 1] = points[:, 1] + xy[1]
        return points

    def get_local_map(self, global_map):
        half_size = max(global_map.shape[0], global_map.shape[1])
        ego_map = np.zeros((half_size * 2, half_size * 2, 3), dtype = np.uint8)
        xy = self.data.qpos[:2]
        ori = self.data.qpos[2]
        x_img = int(-xy[1] / self.resolution)
        y_img = int(-xy[0] / self.resolution)
        x_img -= int(np.floor(self.allo_map_side_range[0] / self.resolution))
        y_img += int(np.ceil(self.allo_map_fwd_range[1] / self.resolution))
        x_start = half_size - y_img
        x_end = x_start + global_map.shape[1]
        y_start = half_size - x_img
        y_end = y_start + global_map.shape[0]
        assert x_start >= 0 and y_start >= 0 and \
               x_end <= ego_map.shape[0] and y_end <= ego_map.shape[1]
        ego_map[x_start: x_end, y_start: y_end] = global_map
        center = (half_size, half_size)
        ori = -ori
        if ori < np.pi:
            ori += 2 * np.pi
        elif ori > np.pi:
            ori -= 2 * np.pi
        ori = 180 * ori / np.pi
        M = cv2.getRotationMatrix2D(center, ori, 1.0)
        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        ego_map[half_size, half_size] = 255
        ego_map[half_size + 1, half_size] = 255
        ego_map[half_size, half_size + 1] = 255
        ego_map[half_size - 1, half_size] = 255
        ego_map[half_size, half_size - 1] = 255
        start = half_size - int(30 / self.resolution)
        end = half_size + int(30 / self.resolution)
        loc_map = ego_map[start: end, start: end]
        return loc_map

    def get_maps(self,
            depth,
            rgb,
            res = 0.1,
            side_range = (-40, 40),
            fwd_range = (-40, 40),
            height_range = (0, 1.5)
        ):
        borders_cloud, floor_cloud, objects_cloud = self.get_ego_clouds(depth, rgb)

        border_x_points = borders_cloud[:, 0]
        border_y_points = borders_cloud[:, 1]
        border_z_points = borders_cloud[:, 2]

        border_x_img = (-border_y_points / res).astype(np.int32)
        border_y_img = (-border_x_points / res).astype(np.int32)
        
        border_x_img -= int(np.floor(side_range[0] / res))
        border_y_img += int(np.ceil(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        border_pixel_values = np.clip(a=border_z_points,
                               a_min=height_range[0],
                               a_max=height_range[1])

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        border_pixel_values = scale_to_255(border_pixel_values,
                                    minimum=height_range[0],
                                    maximum=height_range[1])

        border_x_img, border_y_img, border_pixel_values = sort_arrays(
            border_x_img,
            border_y_img,
            border_pixel_values
        )

        self.map[border_y_img, border_x_img, 0] = border_pixel_values

        floor_x_points = floor_cloud[:, 0]
        floor_y_points = floor_cloud[:, 1]
        floor_z_points = floor_cloud[:, 2]

        floor_x_img = (-floor_y_points / res).astype(np.int32)
        floor_y_img = (-floor_x_points / res).astype(np.int32)
        
        floor_x_img -= int(np.floor(side_range[0] / res))
        floor_y_img += int(np.ceil(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        floor_pixel_values = np.clip(a=floor_z_points,
                               a_min=height_range[0],
                               a_max=height_range[1])

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        floor_pixel_values = scale_to_255(floor_pixel_values,
                                    minimum=height_range[0],
                                    maximum=height_range[1])

        floor_x_img, floor_y_img, floor_pixel_values = sort_arrays(
            floor_x_img,
            floor_y_img,
            floor_pixel_values
        )
        #floor_pixel_values[:] = 10
        self.map[floor_y_img, floor_x_img, 1] = floor_pixel_values

        objects_x_points = objects_cloud[:, 0]
        objects_y_points = objects_cloud[:, 1]
        objects_z_points = objects_cloud[:, 2]

        objects_x_img = (-objects_y_points / res).astype(np.int32)
        objects_y_img = (-objects_x_points / res).astype(np.int32)
        
        objects_x_img -= int(np.floor(side_range[0] / res))
        objects_y_img += int(np.ceil(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        objects_pixel_values = np.clip(a=objects_z_points,
                               a_min=height_range[0],
                               a_max=height_range[1])

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        objects_pixel_values = scale_to_255(objects_pixel_values,
                                    minimum=height_range[0],
                                    maximum=height_range[1])

        objects_x_img, objects_y_img, objects_pixel_values = sort_arrays(objects_x_img, objects_y_img, objects_pixel_values)

        self.map[objects_y_img, objects_x_img, 2] = objects_pixel_values

        self.maps.pop(0)
        self.maps.append(self.map.copy())
        # ego_map = self.get_ego_map(borders_cloud, floor_cloud, objects_cloud)
        loc_map = self.get_local_map(self.map)
        return loc_map

    def get_known_blobs(
            self,
            frame: np.ndarray
    ) -> List[np.ndarray]:
        """Detects All known blobs in a given image

        :param frame: Visual Perception Input
        :type frame: np.ndarray
        """
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        """Getter method for current observations.

        The following are the mandatory components of the observation `dict`:
        * Current Visual Observation `frame_t`
        * Current Propreceptive Observation `sensors`
        * Current Estimated Position in global frame of reference `pos`
        * Start Position of the agent in the global frame of reference `start_pos`
        * Information if the target object is present in the frame or not `inframe`

        :return: Current Observations.
        :rtype: Union[np.ndarray, Dict[str, np.ndarray]]
        """
        obs = self.wrapped_env._get_obs()
        # obs['front'] = cv2.resize(obs['front'], (320, 320))
        img = obs['front'].copy()
        # assert img.shape[0] == img.shape[1]
        # Target Detection and Attention Window
        bbx = self.detect_target(img)
        window, bbx = self.get_attention_window(obs['front'], bbx)
        inframe = True if len(bbx) > 0 else False 

        # Sampled Action
        sampled_action = self.get_action().astype(np.float32)
        low = self.action_space.low
        high = self.action_space.high
        scaled_sampled_action = (sampled_action - low) / (high - low)

        # Sensor Readings
        ## Goal
        goal = self._task.objects[self._task.goal_index].pos - self.wrapped_env.get_xy()
        goal = np.array([
            np.linalg.norm(goal) / np.linalg.norm(self._task.objects[self._task.goal_index].pos - self._init_pos),
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
            np.array([self.get_ori()], dtype = np.float32),
            np.array([self.reward]).copy()
        ] + [
            (action.copy() - self.action_space.low) / (self.action_space.high - self.action_space.low) for action in self.actions
        ], -1)

        # complete_ego_map, border_ego_map, floor_ego_map, objects_ego_map, target_ego_map = self.get_ego_maps(obs['front_depth'], obs['front'])
        loc_map = self.get_maps(
            obs['front_depth'],
            obs['front'],
            res = self.resolution,
            side_range=self.allo_map_side_range,
            fwd_range=self.allo_map_fwd_range,
            height_range = self.allo_map_height_range
        )
        if params['debug']:
            size = img.shape[0]
            cv2.imshow('stream camera', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow('stream attention window', cv2.cvtColor(cv2.resize(
                window, (size, size)
            ), cv2.COLOR_RGB2BGR))
            cv2.imshow('depth stream', (obs['front_depth'] - 0.86) / 0.14)
            top = self.render('rgb_array')
            cv2.imshow('position stream', top)
            #cv2.imshow('bird eye view', cv2.resize(bird_eye_view, (image_width, image_height)))
            """
            cv2.imshow('ego map', complete_ego_map)
            cv2.imshow('borders', border_ego_map)
            cv2.imshow('floor', floor_ego_map)
            cv2.imshow('objects', objects_ego_map)
            cv2.imshow('target', target_ego_map)
            """
            cv2.imshow('global egoocentric map', cv2.resize(loc_map[:, :, :3], (image_width, image_height)))
            cv2.imshow('previous global egocentric map', cv2.resize(self.loc_map[0][:, :, :3], (image_width, image_height)))
            #cv2.imshow('egocentric map', cv2.resize(ego_map, (image_width, image_height)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass

        shape = window.shape[:2]
        scale_2 = cv2.resize(obs['front'], shape)
        depth = np.expand_dims(
            cv2.resize(
                obs['front_depth'].copy(), shape
            ), 0
        )

        positions = np.concatenate(self.positions + [self._task.objects[self._task.goal_index].pos], -1)

        _obs = {
            'scale_1' : window.copy(),
            'scale_2' : scale_2.copy(),
            'sensors' : sensors,
            'sampled_action' : sampled_action.copy(),
            'scaled_sampled_action' : scaled_sampled_action.copy(),
            'depth' : depth,
            'inframe' : np.array([inframe], dtype = np.float32),
            'positions' : positions.copy(),
            'loc_map' : loc_map.copy(),
            'prev_loc_map' : self.loc_map[0].copy(),
            'bbx' : bbx.copy()
        }

        self.loc_map.pop(0)
        self.loc_map.append(loc_map.copy())

        if params['add_ref_scales']:
            ref_scale_1, ref_scale_2 = self.get_scales(obs['front'].copy(), []) 
            ref_scale_2 = cv2.resize(ref_scale_2, shape)
            _obs['ref_scale_1'] = ref_scale_1.copy()
            _obs['ref_scale_2'] = ref_scale_2.copy()

        return _obs

    def reset(self) -> np.ndarray:
        self.set_structure()
        self.collision_count = 0
        self.t = 0
        self.total_eps += 1
        self.close()
        self.set_env()
        action = self.actions[0]
        self.actions = [np.zeros_like(action) for i in range(self.n_steps)]
        goal = self._task.objects[self._task.goal_index].pos - self.wrapped_env.get_xy()
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

    """
    def _render_image(self) -> np.ndarray:
        self._mj_offscreen_viewer._set_mujoco_buffers()
        self._mj_offscreen_viewer.render(*self._image_shape)
        pixels = self._mj_offscreen_viewer.read_pixels(*self._image_shape, depth=False)
        return np.asarray(pixels[::-1, :, :], dtype=np.uint8)
    """

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
        return collision, blind, outbound

    def conditional_blind(self, obs, yaw, b):
        penalty = 0.0
        for direction in b:
            if direction:
                penalty += -0.005 * self._inner_reward_scaling
        
        if self._is_in_collision():
            penalty += -0.99 * self._inner_reward_scaling
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
        last_coverage = self.get_coverage(self.maps[0])
        coverage = self.get_coverage(self.maps[-1])
        coverage_reward = (coverage - last_coverage) * 0.05

        # Computing the reward in "https://ieeexplore.ieee.org/document/8398461"
        goal = self._task.objects[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals.pop(0)
        self.goals.append(goal)
        self.positions.pop(0)
        self.positions.append(self.data.qpos.copy())
        theta_t = self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori())
        qvel = self.wrapped_env.data.qvel.copy()
        vyaw = qvel[self.wrapped_env.ORI_IND]
        vmax = self.target_speed
        if bool(next_obs['inframe'][0]):
            inner_reward = (v / vmax) * np.cos(theta_t) * (1 - (np.abs(vyaw) / params['max_vyaw'])) * 0.005
            inner_reward = self._inner_reward_scaling * inner_reward

        # Task Reward Computation
        outer_reward = 0
        outer_reward = self._task.reward(next_pos, bool(next_obs['inframe'][0]), self._start_pos) * 0.005
        done = self._task.termination(self.wrapped_env.get_xy(),  bool(next_obs['inframe'][0]))
        info["position"] = self.wrapped_env.get_xy()

        # Collision Penalty Computation
        index = self._get_current_cell()
        self._current_cell = index
        almost_collision, blind, outbound = self.check_position(next_pos)
        if almost_collision:
            collision_penalty += -0.005 * self._inner_reward_scaling
        next_obs, penalty = self.conditional_blind(next_obs, yaw, blind)
        collision_penalty += penalty

        # Reward and Info Declaration
        if done:
            outer_reward += 1.0
            info['is_success'] = True
        else:
            info['is_success'] = False
        if outbound:
            collision_penalty += -0.05 * self._inner_reward_scaling
            next_obs['scale_1'] = np.zeros_like(next_obs['scale_1'])
            next_obs['scale_2'] = np.zeros_like(next_obs['scale_2'])
            done = True
        if self.t > self.max_episode_size:
            done = True
        
        if collision_penalty < -0.05:
            done = True

        reward = inner_reward + outer_reward + collision_penalty + coverage_reward
        self.reward = reward
        info['inner_reward'] = inner_reward
        info['outer_reward'] = outer_reward
        info['collision_penalty'] = collision_penalty
        info['coverage_reward'] = coverage_reward
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
        for i, goal in enumerate(self._task.objects):
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

        return np.rot90(np.flipud(img))


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
        self.target_speed = 4

    def _set_action_space(self):
        """Set class attribute `_action_space`.
        """
        self._action_space = gym.spaces.MultiDiscrete([6, 11])

    def discrete_v(self, v):
        """Discretize velocity.
        :param v: Input Velocity
        :type v: Union[float, np.ndarray]
        :return: Discretized Velocity
        :rtype: float
        """
        if v < self.target_speed / 8:
            v = 4
        else:
            v = 0
        return v

    def discrete_vyaw(self, vyaw):
        """Discretize yaw velocity.
        :param v: Input Velocity
        :type v: Union[float, np.ndarray]
        :return: Discretized Velocity
        :rtype: float
        """
        if vyaw < -1.125:
            vyaw = 0
        elif vyaw < -0.5625:
            vyaw = 1
        elif vyaw < -0.28125:
            vyaw = 2
        elif vyaw < -0.140625:
            vyaw = 3
        elif vyaw < -0.046875:
            vyaw = 4
        elif vyaw < 0.046875:
            vyaw = 5
        elif vyaw < 0.140625:
            vyaw = 6
        elif vyaw < 0.28125:
            vyaw = 7
        elif vyaw < 0.5625:
            vyaw = 8
        elif vyaw < 1.125:
            vyaw = 9
        else:
            vyaw = 10
        return vyaw

    def get_action(self):
        """Sample appropriate action from the predefined algorithm.
        :return: Sampled Action
        :rtype: np.ndarray
        """
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
        v = self.discrete_v(v)
        self.sampled_action = np.array([
            0,
            vyaw,
        ], dtype = np.float32)
        return self.sampled_action

    def continuous_action(self, action):
        """Convert discrete action to continuous action.
        :param action: Discrete action
        :type action: np.ndarray
        :return: Continunous action
        :rtype: np.ndarraiy
        :raises ValueError: Error raised when discrete actions are out of bounds.
        """
        move, omega = action
        if move == 0:
            move = self.target_speed
        elif move == 1:
            move = self.target_speed / 8
        elif move == 2:
            move = self.target_speed / 4
        elif move == 3:
            move = self.target_speed / 2
        elif move == 4:
            move = 0
        elif move == 5:
            move = -self.target_speed / 4
        else:
            raise ValueError

        if omega == 0:
            omega = -1.5
        elif omega == 1:
            omega = -0.75
        elif omega == 2:
            omega = -0.375
        elif omega == 3:
            omega = -0.1875
        elif omega == 4:
            omega = -0.092375
        elif omega == 5:
            omega = 0.0
        elif omega == 6:
            omega = 0.092375
        elif omega == 7:
            omega = 0.1875
        elif omega == 8:
            omega = 0.375
        elif omega == 9:
            omega = 0.75
        elif omega == 10:
            omega = 1.5
        else:
            raise ValueError

        action = np.array([move, omega], dtype = np.float32)
        return action

    def _get_obs(self) -> np.ndarray:
        """Internal method for processing the observation for downstream tasks. Needs to be reimplemented for every new environment.
        :return: Environment Observation.
        :rtype: Union[Dict[str, np.ndarray], np.ndarray]
        """
        obs = self.wrapped_env._get_obs()
        #obs['front'] = cv2.resize(obs['front'], (320, 320))
        img = obs['front'].copy()
        #assert img.shape[0] == img.shape[1]
        # Target Detection and Attention Window
        bbx = self.detect_target(img)
        inframe = True if len(bbx) > 0 else False 
        window, bbx = self.get_attention_window(obs['front'], bbx)

        # Sampled Action
        sampled_action = self.get_action().astype(np.float32)

        # Sensor Readings
        ## Goal
        goal = self._task.objects[self._task.goal_index].pos - self.wrapped_env.get_xy()
        goal = np.array([
            np.linalg.norm(goal) / np.linalg.norm(self._task.objects[self._task.goal_index].pos - self._init_pos),
            self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori()) / np.pi
        ], dtype = np.float32)
        ## Velocity
        max_vel = np.array([
            self.wrapped_env.VELOCITY_LIMITS,
            self.wrapped_env.VELOCITY_LIMITS,
            self.wrapped_env.action_space.high[1]
        ])
        high = self.wrapped_env.action_space.high[1]
        low = self.wrapped_env.action_space.low[1]
        sensors = np.concatenate([
            self.data.qvel.copy() / max_vel,
            np.array([self.get_ori() / np.pi], dtype = np.float32),
            np.array([self.reward]).copy()
        ] + [
            (action.copy() - low) / (high - low) for action in self.actions
        ], -1)

        #complete_ego_map, border_ego_map, floor_ego_map, objects_ego_map, target_ego_map = self.get_ego_maps(obs['front_depth'], obs['front'])
        loc_map = self.get_maps(
            obs['front_depth'],
            obs['front'],
            res = self.resolution,
            side_range=self.allo_map_side_range,
            fwd_range=self.allo_map_fwd_range,
            height_range = self.allo_map_height_range
        )

        """DEBUGGING CODE. NEEDS TO BE REMOVED.
        """
        if params['debug']:
            size = img.shape[0]
            cv2.imshow('stream camera', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow('stream attention window', cv2.cvtColor(cv2.resize(
                window, (size, size)
            ), cv2.COLOR_RGB2BGR))
            cv2.imshow('depth stream', (obs['front_depth'] - 0.86) / 0.14)
            top = self.render('rgb_array')
            cv2.imshow('position stream', top)
            # cv2.imshow('bird eye view', cv2.resize(bird_eye_view, (image_width, image_height)))
            """
            cv2.imshow('ego map', complete_ego_map)
            cv2.imshow('borders', border_ego_map)
            cv2.imshow('floor', floor_ego_map)
            cv2.imshow('objects', objects_ego_map)
            cv2.imshow('target', target_ego_map)
            """
            cv2.imshow('global egoocentric map', cv2.resize(loc_map[:, :, :3], (image_width, image_height)))
            cv2.imshow('previous global egocentric map', cv2.resize(self.loc_map[0][:, :, :3], (image_width, image_height)))
            # cv2.imshow('egocentric map', cv2.resize(ego_map, (image_width, image_height)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass

        shape = window.shape[:2]
        scale_2 = cv2.resize(obs['front'], shape)
        depth = np.expand_dims(
            cv2.resize(
                obs['front_depth'].copy(), shape
            ), 0
        )

        positions = np.concatenate(self.positions + [self._task.objects[self._task.goal_index].pos], -1)

        _obs = {
            'scale_1': window.copy(),
            'scale_2': scale_2.copy(),
            'sensors': sensors,
            'sampled_action': sampled_action.copy(),
            'scaled_sampled_action': sampled_action.copy(),
            'depth': depth,
            'inframe': np.array([inframe], dtype=np.float32),
            'positions': positions.copy(),
            'loc_map': loc_map.copy(),
            'prev_loc_map': self.loc_map[0].copy(),
            'bbx': bbx.copy()
        }

        self.loc_map.pop(0)
        self.loc_map.append(loc_map.copy())

        if params['add_ref_scales']:
            ref_scale_1, ref_scale_2 = self.get_scales(obs['front'].copy(), []) 
            ref_scale_2 = cv2.resize(ref_scale_2, shape)
            _obs['ref_scale_1'] = ref_scale_1.copy()
            _obs['ref_scale_2'] = ref_scale_2.copy()

        return _obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        """
        # Proprocessing and Environment Update
        assert self.action_space.contains(action)
        action = self.continuous_action(action)
        self.t += 1
        self.total_steps += 1
        info = {}
        self.actions.pop(0)
        self.actions.append(action.copy())
        _, inner_reward, _, info = self.wrapped_env.step(action)

        # Observation and Parameter Gathering
        x, y = self.wrapped_env.get_xy()
        yaw = self.get_ori()
        v = np.linalg.norm(self.data.qvel[:2])
        self.state.set(x, y, v, yaw)
        next_pos = self.wrapped_env.get_xy()
        collision_penalty = 0.0
        next_obs = self._get_obs()
        last_coverage = self.get_coverage(self.maps[0])
        coverage = self.get_coverage(self.maps[-1])
        coverage_reward = (coverage - last_coverage) * 0.05

        # Computing the reward in "https://ieeexplore.ieee.org/document/8398461"
        goal = self._task.objects[self._task.goal_index].pos - self.wrapped_env.get_xy()
        self.goals.pop(0)
        self.goals.append(goal)
        self.positions.pop(0)
        self.positions.append(self.data.qpos.copy())
        theta_t = self.check_angle(np.arctan2(goal[1], goal[0]) - self.get_ori())
        qvel = self.wrapped_env.data.qvel.copy()
        vyaw = qvel[self.wrapped_env.ORI_IND]
        vmax = self.target_speed
        if bool(next_obs['inframe'][0]):
            inner_reward = (v / vmax) * np.cos(theta_t) * (1 - (np.abs(vyaw) / params['max_vyaw'])) * 0.005
            inner_reward = self._inner_reward_scaling * inner_reward

        # Task Reward Computation
        outer_reward = 0
        outer_reward = self._task.reward(next_pos, bool(next_obs['inframe'][0]), self._start_pos) * 0.005
        done = self._task.termination(self.wrapped_env.get_xy(),  bool(next_obs['inframe'][0]))
        info["position"] = self.wrapped_env.get_xy()

        # Collision Penalty Computation
        index = self._get_current_cell()
        self._current_cell = index
        almost_collision, blind, outbound = self.check_position(next_pos)
        if almost_collision:
            collision_penalty += -0.005 * self._inner_reward_scaling
        next_obs, penalty = self.conditional_blind(next_obs, yaw, blind)
        collision_penalty += penalty

        # Reward and Info Declaration
        if done:
            outer_reward += 1.0
            info['is_success'] = True
        else:
            info['is_success'] = False
        if outbound:
            collision_penalty += -0.05 * self._inner_reward_scaling
            next_obs['scale_1'] = np.zeros_like(next_obs['scale_1'])
            next_obs['scale_2'] = np.zeros_like(next_obs['scale_2'])
            done = True
        if self.t > self.max_episode_size:
            done = True
        
        if collision_penalty < -0.05:
            done = True

        reward = inner_reward + outer_reward + collision_penalty + coverage_reward

        self.reward = reward
        info['inner_reward'] = inner_reward
        info['outer_reward'] = outer_reward
        info['collision_penalty'] = collision_penalty
        info['coverage_reward'] = coverage_reward
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

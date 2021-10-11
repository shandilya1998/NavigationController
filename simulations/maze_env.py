"""
Mujoco Maze environment.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple, Type
import gym
import numpy as np
import networkx as nx
from simulations import maze_env_utils, maze_task
from simulations.agent_model import AgentModel
from utils.env_utils import convert_observation_to_space, \
    quintic_polynomials_planner, proportional_control, \
    pure_pursuit_steer_control, State, TargetCourse, States
import random
import copy
from constants import params

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.join(os.getcwd(), 'assets', 'xml')

class MazeEnv(gym.Env):
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
        include_position: bool = True,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        inner_reward_scaling: float = 1.0,
        restitution_coef: float = 0.8,
        task_kwargs: dict = {},
        websock_port: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.t = 0  # time steps
        self._task = maze_task(maze_size_scaling, **task_kwargs)
        self._maze_height = height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self._inner_reward_scaling = inner_reward_scaling
        self._put_spin_near_agent = self._task.PUT_SPIN_NEAR_AGENT
        # Observe other objectives
        self._restitution_coef = restitution_coef

        self._maze_structure = structure = self._task.create_maze()
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

        if model_cls.MANUAL_COLLISION:
            if model_cls.RADIUS is None:
                raise ValueError("Manual collision needs radius of the model")
            self._collision = maze_env_utils.CollisionDetector(
                structure,
                size_scaling,
                torso_x,
                torso_y,
                model_cls.RADIUS,
            )
            # Now all object balls have size=1.0
            self._objball_collision = maze_env_utils.CollisionDetector(
                structure,
                size_scaling,
                torso_x,
                torso_y,
                self._task.OBJECT_BALL_SIZE,
            )
        else:
            self._collision = None

        def func(x):
            x_int, x_frac = int(x), x % 1
            if x_frac > 0.5:
                x_int += 1
            return x_int

        self._xy_to_rowcol = lambda x, y: (
            func((y + torso_y) / size_scaling),
            func((x + torso_x) / size_scaling),
        )
        self._rowcol_to_xy = lambda r, c: (
            c * size_scaling - torso_x,
            r * size_scaling - torso_y
        )
        # walls (immovable), chasms (fall), movable blocks
        self._view = np.zeros([5, 5, 3])

        # Let's create MuJoCo XML
        xml_path = os.path.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        height_offset = 0.0
        if self.elevated:
            # Increase initial z-pos of ant.
            height_offset = height * size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set("pos", f"0 0 {0.75 + height_offset:.2f}")
        if self.blocks:
            # If there are movable blocks, change simulation settings to perform
            # better contact detection.
            default = tree.find(".//default")
            default.find(".//geom").set("solimp", ".995 .995 .01")

        self.movable_blocks = []
        self.object_balls = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]
                if struct.is_robot() and self._put_spin_near_agent:
                    struct = maze_env_utils.MazeCell.SPIN
                x, y = j * size_scaling - torso_x, i * size_scaling - torso_y
                h = height / 2 * size_scaling
                size = size_scaling * 0.5
                if self.elevated and not struct.is_chasm():
                    # Create elevated platform.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"elevated_{i}_{j}",
                        pos=f"{x} {y} {h}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1",
                    )
                if struct.is_block():
                    # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {h + height_offset}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                elif struct.can_move():
                    # Movable block.
                    self.movable_blocks.append(f"movable_{i}_{j}")
                    _add_movable_block(
                        worldbody,
                        struct,
                        i,
                        j,
                        size_scaling,
                        x,
                        y,
                        h,
                        height_offset,
                    )
                elif struct.is_object_ball():
                    # Movable Ball
                    self.object_balls.append(f"objball_{i}_{j}")
                    _add_object_ball(worldbody, i, j, x, y, self._task.OBJECT_BALL_SIZE)

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if "name" not in geom.attrib:
                raise Exception("Every geom of the torso must have a name")

        # Set goals
        for i, goal in enumerate(self._task.goals):
            z = goal.pos[2] if goal.dim >= 3 else 0.1 *  maze_size_scaling
            if goal.custom_size is None:
                size = f"{maze_size_scaling * 0.1}"
            else:
                size = f"{goal.custom_size}"
            ET.SubElement(
                worldbody,
                "site",
                name=f"goal_site{i}",
                pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                size=f"{maze_size_scaling * 0.1}",
                rgba=goal.rgb.rgba_str(),
            )
        
        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)
        self.world_tree = tree
        self.wrapped_env = model_cls(file_path=file_path, **kwargs)
        ob = self._get_obs()
        self._set_observation_space(ob)
        self._websock_port = websock_port
        self._mj_offscreen_viewer = None
        self._websock_server_pipe = None
        self.__create_maze_graph()
        self.sampled_path = self.__sample_path()
        self._previous_cell = -1
        self._current_cell = copy.deepcopy(self.sampled_path[0])
        self._previous_cell = copy.deepcopy(self._current_cell)
        self._next_cell = self.sampled_path[1]
        self.cells_passed = 0
        self.__setup_vel_control()

    def __setup_vel_control(self):
        self.count = 0
        self.cells_passed += 1
        next_cell_row, next_cell_col = self._graph_to_structure_index(
            self._next_cell
        )
        self.__next_cell_x, self.__next_cell_y = self._rowcol_to_xy(
            next_cell_row, next_cell_col
        )
        next_index = self.sampled_path.index(self._next_cell)
        if self.cells_passed == 1:
            sv = 0.0  # start speed [m/s]
            sa = 0.0  # start accel [m/ss]
        else:
            sv = copy.deepcopy(self.gv)
            sa = copy.deepcopy(self.ga)
        self.gyaw = 0.0
        self.gv = 0.0
        self.ga = 0.0
        
        if not next_index == len(self.sampled_path) - 1:
            next_next_index = self.sampled_path[next_index + 1]
            next_next_row, next_next_col = self._graph_to_structure_index(
                next_next_index
            )
            next_next_x, next_next_y = self._rowcol_to_xy(
                next_next_row, next_next_col
            )
            self.gyaw = np.arctan2(
                self.__next_cell_y - next_next_y,
                self.__next_cell_x - next_next_x
            )
            self.gv = np.random.uniform(
                low = 0.0,
                high = self.wrapped_env.VELOCITY_LIMITS
            )
            self.ga = 1.0
        self.__init_robot_x, self.__init_robot_y = self.wrapped_env.get_xy()
        self.__init_robot_ori = self.wrapped_env.get_ori()
        if self.__init_robot_ori > np.pi:
            self.__init_robot_ori -= 2 * np.pi
        sx = self.__init_robot_x  # start x position [m]
        sy = self.__init_robot_y  # start y position [m]
        syaw = self.__init_robot_ori  # start yaw angle [rad]
        gx = self.__next_cell_x  # goal x position [m]
        gy = self.__next_cell_y  # goal y position [m]
        max_accel = 1000.0  # max accel [m/ss]
        max_jerk = 100.0  # max jerk [m/sss]
        dt = self.wrapped_env.dt  # time tick [s]
        self.time, self.x, self.y, \
            self.yaw, self.v, self.vx, self.vy, \
            self.target_a, self.j = quintic_polynomials_planner(
                sx, sy, syaw,
                sv, sa, gx,
                gy, self.gyaw, self.gv,
                self.ga, max_accel,
                max_jerk, dt
            )
        self.vyaw = np.gradient(
            self.yaw,
            self.time ,
            axis = 0)
        self.vyaw[0] = 0
        self.vyaw[1] = self.vyaw[2] / 2
        self.state = State(
            x = self.wrapped_env.sim.data.qpos[0],
            y = self.wrapped_env.sim.data.qpos[1],
            yaw = self.wrapped_env.sim.data.qpos[2],
            v = np.sqrt(
                np.square(
                    self.wrapped_env.sim.data.qvel[0]
                ) + np.square(
                    self.wrapped_env.sim.data.qvel[1]
                )
            ),
            WB = 1.0 * self._maze_size_scaling
        )
        self.lastIndex = len(self.x) - 1
        self.states = States()
        self.states.append(self.count * self.wrapped_env.dt, self.state)
        self.target_course = TargetCourse(self.x, self.y)
        self.target_ind, _ = self.target_course.search_target_index(self.state)
        self.__init_distance = np.array([
            np.abs(self.__next_cell_x - self.__init_robot_x),
            np.abs(self.__next_cell_y - self.__init_robot_y)
        ])
        self.__init_ori = (np.arctan2(
            self.__next_cell_y - self.__init_robot_y,
            self.__next_cell_x - self.__init_robot_x
        ) - self.__init_robot_ori + np.pi) % (2 * np.pi) - np.pi

    def __sample_path(self):
        robot_x, robot_y = self.wrapped_env.get_xy()
        robot_ori = self.wrapped_env.get_ori()
        row, col = self._xy_to_rowcol(robot_x, robot_y)
        source = self._structure_to_graph_index(row, col)
        goal_pos = self._task.goals[0].pos[:2]
        row, col = self._xy_to_rowcol(goal_pos[0], goal_pos[1])
        target = self._structure_to_graph_index(row, col)
        paths = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(
            self._maze_graph,
            source,
            target
        ))
        return random.choice(paths)

    def get_action(self):
        robot_x, robot_y = self.wrapped_env.get_xy()
        robot_ori = self.wrapped_env.get_ori()
        robot_ori = robot_ori % (2 * np.pi)
        if robot_ori > np.pi:
            robot_ori -= 2 * np.pi
        
        """
        dist = np.array([
            np.abs(robot_x - self.__next_cell_x),
            np.abs(robot_y - self.__next_cell_y)
        ])  
        ori = np.arctan2(
            robot_y - self.__next_cell_y,
            robot_x - self.__next_cell_x
        )   
        acc = (0.5 - np.divide(
            dist,
            self.__init_distance,
            out = np.zeros_like(dist),
            where = self.__init_distance != 0
        ))
        self.sampled_action[:2] = \
            self.sampled_action[:2] + acc * self.wrapped_env.dt
        acc_w = (0.5 - np.divide(
            ori,
            self.__init_ori,
            out = np.zeros_like(self.__init_ori),
            where = self.__init_ori != 0))
        self.sampled_action[2] = self.sampled_action[2] + \
            self.sampled_action[2] * self.wrapped_env.dt
        """
        ai = proportional_control(self.v[self.count], self.state.v)
        di, self.target_ind = pure_pursuit_steer_control(
            self.state, self.target_course, self.target_ind, self.state.WB
        )
        self.state.update(ai, di)
        self.count += 1
        vx = self.state.v * np.cos(self.state.yaw)
        vy = self.state.v * np.sin(self.state.yaw)
        vyaw = self.state.v / self.state.WB * np.tan(di)
        self.sampled_action = np.array([
            vx,
            vy,
            vyaw
        ])
        return self.sampled_action

    def _graph_to_structure_index(self, index):
        row = int(index / len(self._maze_structure))
        col = index % len(self._maze_structure[0])
        return row, col

    def _structure_to_graph_index(self, row, col):
        return row * len(self._maze_structure) + col

    def __check_structure_index_validity(self, i, j):
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

    def __add_edges_to_maze_graph(self, node):
        neighbors = [
            (node['row'] - 1, node['col']),
            (node['row'], node['col'] - 1),
            (node['row'] + 1, node['col']),
            (node['row'], node['col'] + 1),
            (node['row'] + 1, node['col'] - 1),
            (node['row'] - 1, node['col'] + 1),
            (node['row'] + 1, node['col'] + 1),
            (node['row'] - 1, node['col'] - 1)
        ]
        for neighbor in neighbors:
            if self.__check_structure_index_validity(
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

    def __create_maze_graph(self):
        num_row = len(self._maze_structure)
        num_col = len(self._maze_structure[0])
        num_vertices = num_row * num_col
        self._maze_graph = nx.DiGraph()
        self._maze_graph.add_nodes_from(np.arange(
            0, len(self._maze_structure) * len(self._maze_structure[0])
        ))
        for i in range(num_col):
            for j in range(num_row):
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

        for i in range(num_col):
            for j in range(num_row):
                self.__add_edges_to_maze_graph(self._maze_graph.nodes[
                    self._structure_to_graph_index(i, j)
                ])

    def get_ori(self) -> float:
        return self.wrapped_env.get_ori()

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
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

    def _get_obs(self) -> np.ndarray:
        return self.wrapped_env._get_obs()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.wrapped_env.reset()
        # Samples a new start position
        if len(self._init_positions) > 1:
            xy = np.random.choice(self._init_positions)
            self.wrapped_env.set_xy(xy)
        self.sampled_path = self.__sample_path()
        self._previous_cell = -1
        self._current_cell = copy.deepcopy(self.sampled_path[0])
        self._previous_cell = copy.deepcopy(self._current_cell)
        self._next_cell = self.sampled_path[1]
        self.__setup_vel_control()
        return self._get_obs()

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

    def render(self, mode="human", **kwargs) -> Optional[np.ndarray]:
        if mode == "human" and self._websock_port is not None:
            if self._mj_offscreen_viewer is None:
                from mujoco_py import MjRenderContextOffscreen as MjRCO

                from mujoco_maze.websock_viewer import start_server

                self._mj_offscreen_viewer = MjRCO(self.wrapped_env.sim)
                self._websock_server_pipe = start_server(self._websock_port)
            self._websock_server_pipe.send(self._render_image())
        else:
            return self.wrapped_env.render(mode, **kwargs)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

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

    def _objball_positions(self) -> None:
        return [
            self.wrapped_env.get_body_com(name)[:2].copy() for name in self.object_balls
        ]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.t += 1
        if self.wrapped_env.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            old_objballs = self._objball_positions()
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            new_objballs = self._objball_positions()
            # Checks that the new_position is in the wall
            collision = self._collision.detect(old_pos, new_pos)
            if collision is not None:
                pos = collision.point + self._restitution_coef * collision.rest()
                if self._collision.detect(old_pos, pos) is not None:
                    # If pos is also not in the wall, we give up computing the position
                    self.wrapped_env.set_xy(old_pos)
                else:
                    self.wrapped_env.set_xy(pos)
            # Do the same check for object balls
            for name, old, new in zip(self.object_balls, old_objballs, new_objballs):
                collision = self._objball_collision.detect(old, new)
                if collision is not None:
                    pos = collision.point + self._restitution_coef * collision.rest()
                    if self._objball_collision.detect(old, pos) is not None:
                        pos = old
                    idx = self.wrapped_env.model.body_name2id(name)
                    self.wrapped_env.data.xipos[idx][:2] = pos
        else:
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        inner_reward = self._inner_reward_scaling * inner_reward
        outer_reward = self._task.reward(next_obs)
        done = False#self._task.termination(next_obs)
        info["position"] = self.wrapped_env.get_xy()
        index = self.__get_current_cell()  
        if self.count == len(self.time) - 1:
            self._previous_cell = copy.deepcopy(self._current_cell)
            self._current_cell = copy.deepcopy(index)
            if not self.sampled_path.index(self._next_cell) == \
                len(self.sampled_path) - 1 and \
                not index == self.sampled_path[-1]:
                self._next_cell = self.sampled_path[
                    self.sampled_path.index(self._next_cell) + 1
                ]
            elif self._current_cell in self.sampled_path:
                self._next_cell = copy.deepcopy(self._current_cell)
            self.__setup_vel_control()

        if self._current_cell == self.sampled_path[-1]:
            done = True

        return next_obs, inner_reward + outer_reward, done, info

    def __get_current_cell(self):
        robot_x, robot_y = self.wrapped_env.get_xy()
        robot_ori = self.wrapped_env.get_ori()
        row, col = self._xy_to_rowcol(robot_x, robot_y)
        index = self._structure_to_graph_index(row, col)
        return index

    def close(self) -> None:
        self.wrapped_env.close()
        if self._websock_server_pipe is not None:
            self._websock_server_pipe.send(None)


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

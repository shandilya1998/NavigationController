from utils.point_cloud import PointCloudGenerator
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import open3d

env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

pc_gen = PointCloudGenerator(env.sim, min_bound=(-100., -100., -100.), max_bound=(100., 100., 100.))
cloud_with_normals = pc_gen.generateCroppedPointCloud()
world_origin_axes = open3d.geometry.TriangleMesh.create_coordinate_frame()
open3d.visualization.draw_geometries([cloud_with_normals, world_origin_axes])
"""
https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Bird-eye-view.html
"""

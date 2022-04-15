from utils.point_cloud import PointCloudGenerator, point_cloud_2_birdseye
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import open3d
import numpy as np
import matplotlib.pyplot as plt
env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

pc_gen = PointCloudGenerator(env.sim, min_bound=[-35, -35, -0.4], max_bound=[35, 35, 1])
cloud_with_normals = pc_gen.generateCroppedPointCloud(save_img_dir='/home/shandilya/Desktop')
#world_origin_axes = open3d.geometry.TriangleMesh.create_coordinate_frame()
open3d.visualization.draw_geometries([cloud_with_normals])
"""
https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Bird-eye-view.html
"""
cloud = np.asarray(cloud_with_normals.points)
print(cloud[:, 0].min(), cloud[:, 0].max())
print(cloud[:, 1].min(), cloud[:, 1].max())
print(cloud[:, 2].min(), cloud[:, 2].max())

top = env.render('rgb_array')
image = point_cloud_2_birdseye(cloud, res = 0.1, side_range = (-30, 30), fwd_range=(-30, 30), height_range=(-0.4, 1))
#image = image[:-30 * 10, 30 * 5: 30 * 15]
fig, ax = plt.subplots(1, 2, figsize = (10, 5))

print(image.min(), image.max())

ax[0].imshow(top)
ax[1].imshow(image, cmap ='gray')
plt.show()

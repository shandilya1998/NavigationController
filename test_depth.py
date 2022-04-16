from utils.point_cloud import PointCloudGenerator, point_cloud_2_birdseye
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import open3d
import numpy as np
import matplotlib.pyplot as plt
env = MazeEnv(PointEnv, CustomGoalReward4Rooms)

pc_gen = PointCloudGenerator(env.sim, min_bound=[-35, -35, -1], max_bound=[35, 35, 1])
cloud_with_normals, np_cloud = pc_gen.generateCroppedPointCloud(save_img_dir='/home/shandilya/Desktop')
#world_origin_axes = open3d.geometry.TriangleMesh.create_coordinate_frame()
open3d.visualization.draw_geometries([cloud_with_normals, np_cloud])
"""
https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Bird-eye-view.html
"""
cloud = np.asarray(cloud_with_normals.points)
print(cloud.shape)
print(cloud[:, 0].min(), cloud[:, 0].max())
print(cloud[:, 1].min(), cloud[:, 1].max())
print(cloud[:, 2].min(), cloud[:, 2].max())

top = env.render('rgb_array')
image = point_cloud_2_birdseye(cloud, res = 0.1, side_range = (-30, 30), fwd_range=(-30, 30), height_range=(-1, 1))
image = image[:-30 * 10, 30 * 5: 30 * 15]

cloud2 = np.asarray(np_cloud.points)
print(cloud2.shape)
print(cloud2[:, 0].min(), cloud[:, 0].max())
print(cloud2[:, 1].min(), cloud[:, 1].max())
print(cloud2[:, 2].min(), cloud[:, 2].max())

image2 = point_cloud_2_birdseye(cloud2, res = 0.1, side_range = (-30, 30), fwd_range=(-30, 30), height_range=(-1, 1))
print('test')
print(image2.shape)
image2 = image2[:-30 * 10, 30 * 5: 30 * 15]
print(image2.shape)
fig, ax = plt.subplots(1, 3, figsize = (10, 5))


print(np.any(image != image2))
ax[0].imshow(top)
ax[1].imshow(image, cmap ='gray')
ax[2].imshow(image2, cmap = 'gray')
plt.show()

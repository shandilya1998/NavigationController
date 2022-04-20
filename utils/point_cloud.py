#!/usr/bin/env python3

import math
import numpy as np

from PIL import Image as PIL_Image

import open3d as o3d
from constants import image_height, image_width

import numpy as np


"""
http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/
Point Cloud to Bird Eye View
"""
# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================

def sort_arrays(x, y, z):
    indices = z.argsort()
    z = z[indices]
    x = x[indices]
    y = y[indices]
    return x, y, z

def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-10., 10.),  # left-most to right-most
                           fwd_range = (-10., 10.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]#.copy()
    y_points = points[:, 1]#.copy()
    z_points = points[:, 2]#.copy()

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
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    x_img, y_img, pixel_values = sort_arrays(x_img, y_img, pixel_values)

    """
    # FILL PIXEL VALUES IN IMAGE ARRAY 
    for i in range(len(pixel_values)):
        if im[y_img[i], x_img[i]] < pixel_values[i]:
            im[y_img[i], x_img[i]] = pixel_values[i]
    """

    im[y_img, x_img] = pixel_values
    im[y_img + 1, x_img] = pixel_values
    im[y_img, x_img + 1] = pixel_values
    im[y_img - 1, x_img] = pixel_values
    im[y_img, x_img - 1] = pixel_values
    im[y_img + 1, x_img + 1] = pixel_values
    im[y_img - 1, x_img - 1] = pixel_values
    im[y_img + 2, x_img] = pixel_values
    im[y_img, x_img + 2] = pixel_values
    im[y_img - 2, x_img] = pixel_values
    im[y_img, x_img - 2] = pixel_values
    im[y_img + 2, x_img + 2] = pixel_values
    im[y_img - 2, x_img - 2] = pixel_values
    #im[np.logical_and(im < 80, im > 2)] = 0

    return im

"""
Generates numpy rotation matrix from quaternion
@param quat: w-x-y-z quaternion rotation tuple
@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)
@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)
@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix
@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array
@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height
@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels
@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function
    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, min_bound=None, max_bound=None):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # I think these can be set to anything
        self.img_width = image_width
        self.img_height = image_height

        self.cam_names = self.sim.model.camera_names

        self.target_bounds=None
        if min_bound != None and max_bound != None:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # List of camera intrinsic matrices
        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, save_img_dir=None):
        o3d_clouds = []
        np_clouds = []
        cam_poses = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            depth_img = self.captureImage(cam_i)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                depth = (255 * (depth_img - 0.86) / 0.14).astype(np.uint8)
                color_img = self.captureImage(cam_i, False)
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))
                self.saveImg(depth, save_img_dir, 'depth_test_' + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth_img)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
            # Compute world to camera transformation matrix
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id] + np.array([0, 0, 0.5])
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            print(b2w_r)
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            print(c2w)
            transformed_cloud = o3d_cloud.transform(c2w)
            
            points = np.zeros(depth_img.shape + (3,)).reshape(-1, 3)
            depth = depth_img.copy().reshape(-1)
            x = np.repeat(
                np.expand_dims(np.arange(depth_img.shape[0]), 1),
                depth_img.shape[1], 1
            ).reshape(-1)
            y = np.repeat(
                np.expand_dims(np.arange(depth_img.shape[1]), 0),
                depth_img.shape[0], 0
            ).reshape(-1)
            points[:, 1] = (x - self.cam_mats[cam_i][0, 2]) * depth / self.cam_mats[cam_i][0, 0]
            points[:, 0] = (y - self.cam_mats[cam_i][1, 2]) * depth / self.cam_mats[cam_i][1, 1]
            points[:, 2] = depth
            

            """
                https://github.com/deepmind/dm_control/blob/87e046bfeab1d6c1ffb40f9ee2a7459a38778c74/dm_control/mujoco/engine.py#L717
                refer to the above link for more information about the below code
            """
            """
            pos = self.sim.data.cam_xpos[cam_i]
            mat = self.sim.data.cam_xmat[cam_i].reshape(3, 3)
            """
            pos = self.sim.model.body_pos[cam_body_id] + np.array([0, 0, 0.5])
            mat = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            rot_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            mat = np.dot(mat, rot_mat)
            ext = np.eye(4)
            ext[:3, :3] = mat
            ext[:3, 3] = pos

            world_points = np.c_[points, np.ones(len(points))]
            world_points = np.dot(ext, world_points.T).T
            world_points = world_points[:, :3]
            #world_points[:, 2] += 0.5
            """
            for i in range(depth_img.shape[0]):
                for j in range(depth_img.shape[1]):
                    points[i, j, 1] = (i - self.cam_mats[cam_i][0, 2]) * depth_img[i, j] / self.cam_mats[cam_i][0, 0]
                    points[i, j, 0] = (j - self.cam_mats[cam_i][1, 2]) * depth_img[i, j] / self.cam_mats[cam_i][1, 1]
                    points[i, j, 2] = depth_img[i, j]
            """
            pcd = o3d.geometry.PointCloud()
            print(world_points.shape)
            pcd.points = o3d.utility.Vector3dVector(world_points)
            #transformed_pcd = pcd.transform(c2w)
            transformed_pcd = pcd

            # If both minimum and maximum bounds are provided, crop cloud to fit
            #    inside them.
            if self.target_bounds != None:
                transformed_cloud = transformed_cloud.crop(self.target_bounds)
                transformed_pcd = transformed_pcd.crop(self.target_bounds)

            # Estimate normals of cropped cloud, then flip them based on camera
            #    position.
            #transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            #transformed_cloud.orient_normals_towards_camera_location(cam_pos)
            #transformed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            #transformed_pcd.orient_normals_towards_camera_location(cam_pos)
            o3d_clouds.append(transformed_cloud)
            np_clouds.append(transformed_pcd)

        combined_cloud = o3d.geometry.PointCloud()
        np_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        for cloud in np_clouds:
            np_cloud += cloud
        return combined_cloud, np_cloud

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)
            real_depth = self.depthimg2Meters(depth)

            return real_depth
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")

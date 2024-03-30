import numpy as np
from camera import Camera
from giga.perception import *
from giga.utils.transform import Transform
from giga.grasp_sampler import GpgGraspSamplerPcl
import open3d as o3d
import trimesh
import pybullet as p


class GraspSampler:

    def __init__(self, g, c, h, w):
        global get_renders, cam_matrices, height, width
        self.get_renders = g
        self.cam_matrices = c
        self.height = h
        self.width = w

    def sample_grasps(self):
        gripper_finger_depth = 0.05
        size = 6 * gripper_finger_depth
        rgb, depth = self.get_renders(cam_type=Camera.CUSTOMCAM)

        proj_matrix = np.asarray(self.cam_matrices[Camera.CUSTOMCAM][1]).reshape(
            [4, 4], order="F"
        )
        view_matrix = np.asarray(self.cam_matrices[Camera.CUSTOMCAM][0]).reshape(
            [4, 4], order="F"
        )
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1 : 1 : 2 / self.height, -1 : 1 : 2 / self.width]
        y *= -1.0
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3:4]
        points = points[:, :3]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        print("Point cloud size: ", len(point_cloud.points))
        # o3d.visualization.draw_geometries([point_cloud])  # DEBUG
        # filter and downsample points
        point_cloud = point_cloud.voxel_down_sample(0.005)
        # filter to inside the goal volume: sim.target_pose
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            np.array((0.0, -0.65, 1.24)) - np.array([0.30, 0.30, 0.15]),
            np.array((0.0, -0.65, 1.24)) + np.array([0.30, 0.30, 0.15]),
        )
        point_cloud = point_cloud.crop(bounding_box)
        # o3d.visualization.draw_geometries([point_cloud])  # DEBUG

        # sample grasps
        num_grasps = 20
        num_parallel_workers = 4

        sampler = GpgGraspSamplerPcl(
            0.045
        )  # Franka finger depth is actually a little less than 0.05
        safety_dist_above_table = (
            0.005  # tweak based on how high the grasp should be from the table
        )
        print("before sampling")
        grasps, grasps_pos, grasps_rot = sampler.sample_grasps_parallel(
            point_cloud,
            num_parallel=num_parallel_workers,
            num_grasps=num_grasps,
            max_num_samples=200,
            safety_dis_above_table=safety_dist_above_table,
            show_final_grasps=False,
        )
        print("after sampling")
        if len(grasps) == 0:
            return None
        # DEBUG: Visualize grasps:
        # grasps_scene = trimesh.Scene()
        # from giga.utils import visual

        # grasp_mesh_list = [visual.grasp2mesh(g, score=1) for g in grasps]
        # for i, g_mesh in enumerate(grasp_mesh_list):
        #     grasps_scene.add_geometry(g_mesh, node_name=f"grasp_{i}")
        # grasps_scene.show()

        # ToDo: Check grasp definition to execute the grasps
        suitable_grasps = []
        for rot, pos in zip(grasps_rot, grasps_pos):
            rot_mat = np.asarray(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
            grasps_rot = rot_mat @ np.array([0, 0, 1])
            grasps_rot /= np.linalg.norm(grasps_rot)
            #print("Grasp rot: ", grasps_rot)
            if -1.2 < grasps_rot[2] < -0.8:
                if -1 < grasps_rot[1] < 0.5:
                    # if -1.2 < grasps_rot[0] < -0.8:
                    #print("append")
                    suitable_grasps.append((pos, rot))

        if len(suitable_grasps) == 0:
            return None

        average_distance = np.inf
        best_grasp = None
        for pos, rot in suitable_grasps:
            # calc average distance to other grasps
            distance = 0
            for pos2, _ in suitable_grasps:
                distance += np.linalg.norm(pos - pos2)
            distance /= len(suitable_grasps)
            if distance < average_distance:
                average_distance = distance
                best_grasp = (pos, rot)

        return best_grasp[0], best_grasp[1]

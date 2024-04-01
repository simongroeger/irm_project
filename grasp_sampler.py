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
        point_cloud = point_cloud.voxel_down_sample(0.0005)
        # filter to inside the goal volume: sim.target_pose
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            np.array((0.1, -0.65, 1.24)) - np.array([0.30, 0.30, -0.02]),
            np.array((0.1, -0.65, 1.24)) + np.array([0.30, 0.30, 0.40]),
        )
        point_cloud = point_cloud.crop(bounding_box)
        # o3d.visualization.draw_geometries([point_cloud])  # DEBUG

        # sample grasps
        num_grasps = 50
        num_parallel_workers = 4

        sampler = GpgGraspSamplerPcl(
            0.045
        )  # Franka finger depth is actually a little less than 0.05
        safety_dist_above_table = (
            0.00  # tweak based on how high the grasp should be from the table
        )
        print("before sampling")
        grasps, grasps_pos, grasps_rot = sampler.sample_grasps_parallel(
            point_cloud,
            num_parallel=num_parallel_workers,
            num_grasps=num_grasps,
            max_num_samples=500,
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

        # pos filter
        pos_filtered = []
        for rot, pos, grasp in zip(grasps_rot, grasps_pos, grasps):
            if np.linalg.norm(pos - np.array([0, 0, 1.24])) < 0.75:
                pos_filtered.append((pos, rot, grasp))
        orientation_filtered = []
        for pos, rot, grasp in pos_filtered:
            rot_mat = np.asarray(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
            grasps_rot = rot_mat @ np.array([0, 0, -1])
            grasps_rot /= np.linalg.norm(grasps_rot)
            if grasps_rot[1] > 0:
                if grasps_rot[2] > 0:
                    orientation_filtered.append((pos, rot, grasp))
        if len(orientation_filtered) == 0:
            return None

        # grasps_scene = trimesh.Scene()
        # from giga.utils import visual

        # grasp_mesh_list = [
        #     visual.grasp2mesh(g, score=1) for _, _, g in orientation_filtered
        # ]
        # for i, g_mesh in enumerate(grasp_mesh_list):
        #     grasps_scene.add_geometry(g_mesh, node_name=f"grasp_{i}")
        # grasps_scene.show()

        # return best_grasp with the highest angle (z value is the highest)
        best_grasp = None
        z = -np.inf
        for i, (pos, rot, grasp) in enumerate(orientation_filtered):
            rot_mat = np.asarray(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
            grasps_rot = rot_mat @ np.array([0, 0, -1])
            grasps_rot /= np.linalg.norm(grasps_rot)
            if grasps_rot[2] > z:
                z = grasps_rot[2]
                best_grasp = (pos, rot, grasp)

        vec = np.asarray(p.getMatrixFromQuaternion(best_grasp[1])).reshape(
            3, 3
        ) @ np.array([0, 0, -1])
        p.addUserDebugLine(
            best_grasp[0],
            best_grasp[0] + vec,
            [1, 0, 0],
        )
        return best_grasp[0], best_grasp[1]

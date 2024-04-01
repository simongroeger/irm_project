import os
import sys
import glob
import time
import math as m
import numpy as np
from numpy.core.multiarray import array
import pybullet as p
import pybullet_data
from camera import Camera

from typing import Dict, Optional
import matplotlib
from filterpy.kalman import KalmanFilter
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



class RedSphere:
    def __init__(self, pos_px=np.array([0, 0]), pos_cart=np.array([0, 0, 0]), radius=0, amount_px=0):
        self.pos_px = pos_px
        self.pos_cart = pos_cart
        self.radius = radius
        self.amount_px = amount_px

    def __str__(self):
        return "RedSphere  pos: " + str(self.pos_cart) + ", r: " + str(self.radius)

    def get_measurement(self):
        return np.append(self.pos_cart, self.radius)

class ObstacleTracking:

    def __init__(self, get_renders, cam_matrices):
        self.get_renders = get_renders
        self.cam_matrices = cam_matrices

        self.measurement_hz = 40
        
        self.kf: Dict[str, KalmanFilter] = {}
        self.kf["small"] = KalmanFilter(dim_x=4, dim_z = 4)
        self.kf["big"] = KalmanFilter(dim_x=4, dim_z = 4)

        #init kf
        self.init_kf()

        self.da_weight_position = 1
        self.da_weight_size = 1


    def init_kf(self):
        self.kf["small"].x = np.array([0, 0, 0, 0.1])
        self.kf["big"].x = np.array([0, 0, 0, 0.15])

        for key in self.kf:
            self.kf[key].F = np.eye(4)
            self.kf[key].H = np.eye(4)
            self.kf[key].P = 0.2 * np.eye(4)
            self.kf[key].R = 0.05 * np.eye(4)

    def get_obstacles(self):
        return [self.kf["small"].x, self.kf["big"].x]
   
    
    def update_kf(self, ordered_cluster):
        for key in ordered_cluster:
            self.kf[key].predict()
            self.kf[key].update(ordered_cluster[key].get_measurement())

    def is_pixel_red(self, px):
        return (px[0] <= 10 or px[0] >= 160) and px[1] >= 100 and px[1] >= 20

    def detectRedSpheres(self, rgb_img, depth_img, cam_type):
        plt.imsave("custom_img.png", rgb_img[..., :3].astype(np.uint8))

        # detect red spheres: calculate position and size
        hsv = (matplotlib.colors.rgb_to_hsv(rgb_img[..., :3] / 255.0) * 255).astype(np.uint8)

        red_px = []
        adjusted = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3))

        for y in range(rgb_img.shape[0]):
            for x in range(rgb_img.shape[1]):
                h = hsv[y, x, 0]
                if h <= 5:
                    s = hsv[y, x, 1]
                    v = hsv[y, x, 2]
                    if s >= 100 and v >= 20:
                        red_px.append([y, x])
                        adjusted[y, x] = 100
                        #print(y,x,h,s,v)

        if len(red_px) == 0:
            print("no red px found")
            return

        # cluster red pixel
        kmeans = KMeans(n_clusters=2, init='k-means++', algorithm="elkan", random_state=42, n_init=3)
        kmeans.fit(red_px)
        cluster_sizes = np.unique(kmeans.labels_, return_counts=True)[1]

        clusters = []
        for i in range(2):
            y = int(kmeans.cluster_centers_[i][0])
            x = int(kmeans.cluster_centers_[i][1])
            size = cluster_sizes[i]
            if size > 20:
                clusters.append(RedSphere(pos_px=np.array([y,x]).astype(np.uint8), amount_px=size))
                #print(x, y, size)

        # check if path between both cluster centers is red
        if len(clusters) > 1:
            one_cluster = True
            i_min = min(clusters[0].pos_px[0], clusters[1].pos_px[0])
            i_max = max(clusters[0].pos_px[0], clusters[1].pos_px[0])
            j_min = min(clusters[0].pos_px[1], clusters[1].pos_px[1])
            j_max = max(clusters[0].pos_px[1], clusters[1].pos_px[1])
            for i in range(i_min, i_max):
                if not self.is_pixel_red(hsv[i,j_min]):
                    one_cluster = False
            for j in range(j_min, j_max):
                if not self.is_pixel_red(hsv[i_max, j]):
                    one_cluster = False
            
            if one_cluster:
                clusters.append(RedSphere(pos_px=(0.5*clusters[0].pos_px+0.5*clusters[1].pos_px).astype(np.uint8), amount_px=clusters[0].amount_px+clusters[1].amount_px))
                clusters.pop(0)
                clusters.pop(0)

        for cluster in clusters:
            adjusted[cluster.pos_px[0], cluster.pos_px[1]] = 255

        plt.imsave("custom_adjusted_img.png", adjusted[..., :3].astype(np.uint8))

        # compute cartesian coords from pixels
        viewMat, projMat = self.cam_matrices[cam_type]

        v_inv = np.linalg.inv(np.array(viewMat).reshape((4,4)).T)
        p = np.array(projMat).reshape((4,4)).T


        for cluster in clusters:
            px = cluster.pos_px[1]
            py = cluster.pos_px[0]
    
            # from https://stackoverflow.com/questions/70955660/how-to-get-depth-images-from-the-camera-in-pybullet
            # from     depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
            # far and near form simulation.py line 140
            pz = depth_img[py, px]
            z = -0.25 / (5 - 4.95 * pz) 

            p_radius = np.sqrt(cluster.amount_px/np.pi)
            c_radius = 0
            for i in range(10):
                w = p[2,2]*(z - c_radius) + p[2,3]
                c_radius = p_radius / 128.0 * w / p[0,0]
            z -= c_radius

            w = p[2,2]*z + p[2,3]
            x = (px-128) / 128.0 * w / p[0, 0]
            y = -(py-128) / 128.0 * w / p[1, 1]
            p_cam = np.array([x, y, z, 1])
            p_world = v_inv @ p_cam

            cluster.pos_cart = p_world[:3]
            cluster.radius = c_radius

            #print("p", c_radius, p_cam, p_world)
            a = 5

        return clusters


    def da_loss(self, kf, cluster):
        return self.da_weight_position *  np.linalg.norm(kf.x[:3] - cluster.pos_cart) + self.da_weight_size * np.linalg.norm(kf.x[3] - cluster.radius)

    def data_association(self, cart_cluster):
        # data association with loss: distance pos and size
        res = {}

        if len(cart_cluster) == 1:
            loss_a = self.da_loss(self.kf["small"], cart_cluster[0])
            loss_b = self.da_loss(self.kf["big"], cart_cluster[0])

            if loss_a < loss_b:
                res["small"] = cart_cluster[0]
            else:
                res["big"] = cart_cluster[0]

        if len(cart_cluster) == 2:
            loss_0_a = self.da_loss(self.kf["small"], cart_cluster[0]) + self.da_loss(self.kf["big"], cart_cluster[1])
            loss_1_a = self.da_loss(self.kf["small"], cart_cluster[1]) + self.da_loss(self.kf["big"], cart_cluster[0])
            
            if loss_0_a < loss_1_a:
                res["small"] = cart_cluster[0]
                res["big"] = cart_cluster[1]
            else:
                res["small"] = cart_cluster[1]
                res["big"] = cart_cluster[0]

        return res


            
    def step(self):
        cam_type = Camera.FIXEDCAM
        rgb_img, depth_img = self.get_renders(cam_type=cam_type)

        cart_cluster = self.detectRedSpheres(rgb_img, depth_img, cam_type)

        ordered_cluster = self.data_association(cart_cluster)

        #for k in ordered_cluster:
        #    print(k, ordered_cluster[k])

        self.update_kf(ordered_cluster)


        




import os
import sys
import glob
import time
import math as m
import numpy as np
import pybullet as p
import pybullet_data
from enum import Enum
from robot import Robot
from objects import Obstacle, Table, Box, YCBObject, Goal
from pybullet_object_models import ycb_objects
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
from filterpy.kalman import KalmanFilter
from sklearn.cluster import kmeans_plusplus, KMeans
from simulation import Camera



class ObstacleTracking:

    def __init__(self, cam_matrices):
        self.kf_small = KalmanFilter(dim_x=3, dim_z = 4)
        self.kf_big = KalmanFilter(dim_x=3, dim_z = 4)
        self.cam_matrices = cam_matrices


    def step(self, rgb_fixed, depth_fixed, rgb_custom, depth_custom):

        #plt.imsave("custom_img.png", rgb_custom[..., :3].astype(np.uint8))

        # detect red spheres: calculate position and size
        hsv = (matplotlib.colors.rgb_to_hsv(rgb_custom[..., :3] / 255.0) * 255).astype(np.uint8)

        red_px = []
        adjusted = np.zeros((rgb_custom.shape[0], rgb_custom.shape[1], 3))

        for y in range(rgb_custom.shape[0]):
            for x in range(rgb_custom.shape[1]):
                h = hsv[y, x, 0]
                if h <= 10 or h >= 160:
                    s = hsv[y, x, 1]
                    v = hsv[y, x, 2]
                    if s >= 100 and v >= 20:
                        red_px.append([y, x])
                        adjusted[y, x] = 100

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
                clusters.append((y, x, size))
                #print(x, y, size)

        # check if red line between 
        if len(clusters) > 1:
            one_cluster = True
            i = clusters[0][0]
            j = clusters[0][1]
            while i < clusters[1][0] and one_cluster:
                h = hsv[i, j, 0]
                s = hsv[i, j, 1]
                v = hsv[i, j, 2]
                if (h <= 10 or h >= 160) and s >= 100 and v >= 20:
                    i += 1
                    continue
                else:
                    one_cluster = False
                    break
            while i > clusters[1][0] and one_cluster:
                h = hsv[i, j, 0]
                s = hsv[i, j, 1]
                v = hsv[i, j, 2]
                if (h <= 10 or h >= 160) and s >= 100 and v >= 20:
                    i -= 1
                    continue
                else:
                    one_cluster = False
                    break
            while j < clusters[1][1] and one_cluster:
                h = hsv[i, j, 0]
                s = hsv[i, j, 1]
                v = hsv[i, j, 2]
                if (h <= 10 or h >= 160) and s >= 100 and v >= 20:
                    j += 1
                    continue
                else:
                    one_cluster = False
                    break
            while j > clusters[1][1] and one_cluster:
                h = hsv[i, j, 0]
                s = hsv[i, j, 1]
                v = hsv[i, j, 2]
                if (h <= 10 or h >= 160) and s >= 100 and v >= 20:
                    j -= 1
                    continue
                else:
                    one_cluster = False
                    break

            if one_cluster:
                clusters.append((int(0.5*(clusters[0][0]+clusters[1][0])), int(0.5*(clusters[0][1]+clusters[1][1])), clusters[0][2] + clusters[1][2]))
                clusters.pop(0)
                clusters.pop(0)

        for y, x, size in clusters:
            adjusted[y, x] = 255

        #plt.imsave("custom_adjusted_img.png", adjusted[..., :3].astype(np.uint8))

        # compute cartesian coords from pixels
        viewMat, projMat = self.cam_matrices[Camera.CUSTOMCAM]

        v = np.array(viewMat).reshape((4,4)).T
        p = np.array(projMat).reshape((4,4)).T


        cart_cluster = []

        for py, px, size in clusters:

            pz = depth_custom[py, px]
            z = -0.25 / (5 - 4.95 * pz)

            p_radius = np.sqrt(size/np.pi)
            c_radius = 0
            for i in range(3):
                w = p[2,2]*(z - c_radius) + p[2,3]
                c_radius = p_radius / 128.0 * w / p[0,0]
            z -= c_radius

            w = p[2,2]*z + p[2,3]
            x = (px-128) / 128.0 * w / p[0, 0]
            y = -(py-128) / 128.0 * w / p[1, 1]
            p_cam = np.array([x, y, z, 1])
            p_world = np.linalg.inv(v) @ np.array([x, y, z, 1])


            #print("p", c_radius, p_cam, p_world)
            a = 5

            cart_cluster.append((p_world[0], p_world[1], p_world[2], c_radius))
            
        #print()

        # data association with loss: distance pos and size

        # both kf step
        self.kf_small


        return cart_cluster




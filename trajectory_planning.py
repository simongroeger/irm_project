import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import scipy.interpolate 


class TrajectoryPlanning:

    def __init__(self):
        pass

    def plan(self, obstacles):
        
        start = np.array([0.0, -0.65, 1.35])
        goal = np.array([0.6, 0.8, 1.74])

        start1 = np.array(start)
        start1[0] += 0.2
        start1[1] += 0.1

        obstacle1 = obstacles[0][:3]
        obstacle2 = obstacles[1][:3]

        obstacle_middle = 0.5*(obstacle1 + obstacle2)
        obstacle_vertical = obstacle_middle - obstacle1
        obstacle_vertical = np.array([obstacle_vertical[1], -obstacle_vertical[0], 0])
        obstacle_vertical /= np.linalg.norm(obstacle_vertical)
        path1 = obstacle_middle - 0.2 * obstacle_vertical
        path2 = obstacle_middle + 0.2 * obstacle_vertical

        if np.linalg.norm(start - path1) < np.linalg.norm(start - path2):
            trajectory_support_points = np.array([start, start1, obstacle_middle, goal])
        else:
            trajectory_support_points = np.array([start, start1, obstacle_middle, goal])

        b_t, _ = scipy.interpolate.splprep([trajectory_support_points[:, 0], trajectory_support_points[:, 1], trajectory_support_points[:, 2]], k=3, s=10)

        u_new = np.linspace(0, 1, 25)
        values_x, values_y, values_z = scipy.interpolate.splev(u_new, b_t, der=0)

        """
        if np.linalg.norm(obstacle_middle - obstacle1) < 0.5:
            print("wait for suitable trajectory, obstacles are too close")
            return np.array([start, start]), trajectory_support_points


        if np.linalg.norm(goal - obstacle1) < 0.5:
            print("wait for suitable trajectory, obstacle1 is too close to goal")
            return np.array([start, start]), trajectory_support_points


        if np.linalg.norm(goal - obstacle2) < 0.5:
            print("wait for suitable trajectory, obstacle2 is too close to  goal")
            return np.array([start, start]), trajectory_support_points
        """

        for x,y in zip(values_x, values_y):
            if np.sqrt(x*x + y*y) < 0.1:
                print("wait till suitable trajectory, to close to robot base", x, y)
                return np.array([start, start]), trajectory_support_points

        trajectory = np.zeros((len(values_x), 3))
        trajectory[:, 0] = values_x
        trajectory[:, 1] = values_y
        trajectory[:, 2] = 1.5

        return trajectory, trajectory_support_points


        
        




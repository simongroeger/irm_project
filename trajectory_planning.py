import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import scipy.interpolate 


class TrajectoryPlanning:

    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.first_way_point = np.array([0.3, -0.1, 1.74])


    def getNextTarget(self, currentEE,  trajectory):
        if len(trajectory) > 1:
            min_distance = 0
            min_index = -1
            for i, elem in enumerate(trajectory):
                current_distance = np.abs(np.linalg.norm(elem - currentEE))
                if min_index == -1 or current_distance < min_distance:
                    min_index = i
                    min_distance = current_distance

            target_index = min(len(trajectory)-1, min_index + 2)
            return trajectory[target_index]
        else:
            return currentEE
        

    def checkTrajectory(self, trajectory, obstacle1, obstacle2):
        min_distance_obstacle = 0.4
        for p in trajectory:
            if np.linalg.norm(obstacle1 - p) < min_distance_obstacle:
                print("wait for suitable trajectory, obstacle1 is too close to trajectory", p, obstacle1)
                return False

            if np.linalg.norm(obstacle2 - p) < min_distance_obstacle:
                print("wait for suitable trajectory, obstacle2 is too close to trajectory", p, obstacle2)
                return False
            
            if np.linalg.norm(p[:2]) < 0.1:
                print("wait till suitable trajectory, to close to robot base", p)
                return False
        return True
    
    def genTrajectory(self, trajectory_support_points):
        k = 3 if len(trajectory_support_points) > 3 else 2
        b_t, _ = scipy.interpolate.splprep([trajectory_support_points[:, 0], trajectory_support_points[:, 1], trajectory_support_points[:, 2]], k=k, s=3)
        u_new = np.linspace(0, 1, 50)
        values_x, values_y, values_z = scipy.interpolate.splev(u_new, b_t, der=0)

        trajectory = np.zeros((len(values_x), 3))
        trajectory[:, 0] = values_x
        trajectory[:, 1] = values_y
        trajectory[:, 2] = self.goal[2]

        return trajectory

    def plan(self, obstacles):

        obstacle1 = obstacles[0][:3]
        obstacle2 = obstacles[1][:3]

        # direct trajectory
        trajectory_support_points = np.array([self.start, self.first_way_point, self.goal])
        trajectory = self.genTrajectory(trajectory_support_points)

        if self.checkTrajectory(trajectory, obstacle1, obstacle2):
            return True, trajectory, trajectory_support_points
    
        
        obstacle_middle = 0.5*(obstacle1 + obstacle2)
        trajectory_support_points = np.array([self.start, self.first_way_point, obstacle_middle, self.goal])
        trajectory = self.genTrajectory(trajectory_support_points)

        if self.checkTrajectory(trajectory, obstacle1, obstacle2):
            return True, trajectory, trajectory_support_points


        obstacle_vertical = obstacle_middle - obstacle1
        obstacle_vertical = np.array([obstacle_vertical[1], -obstacle_vertical[0], 0])
        obstacle_vertical /= np.linalg.norm(obstacle_vertical)
        path1 = obstacle_middle - 0.2 * obstacle_vertical
        path2 = obstacle_middle + 0.2 * obstacle_vertical

        if np.linalg.norm(self.start - path1) < np.linalg.norm(self.start - path2):
            trajectory_support_points = np.array([self.start, self.first_way_point, path1, obstacle_middle, self.goal])
        else:
            trajectory_support_points = np.array([self.start, self.first_way_point, path2, obstacle_middle, self.goal])

        trajectory = self.genTrajectory(trajectory_support_points)

        if not self.checkTrajectory(trajectory, obstacle1, obstacle2):
            return False, trajectory, trajectory_support_points


        return True, trajectory, trajectory_support_points


        
        




import numpy as np
import matplotlib.pyplot as plt



class Vis:

    def __init__(self, sim_gt_obstacles) -> None:
        self.sim_gt_obstacles = sim_gt_obstacles
        self.fig, self.axs = plt.subplots(1,2)
        self.time_arr = []
        self.plot_arr = []
        for i in range(16):
            self.plot_arr.append([])

        self.fig.suptitle("Object Tracking Error")
        

        self.plot_title = ["o1_x", "o1_y", "o1_z", "o1_size", "o2_x", "o2_y", "o2_z", "o2_size", "kfa_x", "kfa_y", "kfa_z", "kfa_size", "kfb_x", "kfb_y", "kfb_z", "kfb_size"]


    def plot_kf_error(self, time_step, obstacle_tracking):
        global fig, axs, plot_arr

        for ax in self.axs:
            ax.clear()
        
        self.time_arr.append(time_step)

        #for j, obstacle in enumerate(sim.obstacles):
            #   print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)
            #print("kf a", robot.obstacle_tracking.kf["a"].x)
            #print("kf b", robot.obstacle_tracking.kf["b"].x)

        for j, key in enumerate(obstacle_tracking.kf):
            kf = obstacle_tracking.kf[key].x[:3]
            if np.linalg.norm(kf - self.sim_gt_obstacles[0].get_pos()) < np.linalg.norm(kf - self.sim_gt_obstacles[1].get_pos()):
                gt = self.sim_gt_obstacles[0].get_pos()
            else:
                gt = self.sim_gt_obstacles[1].get_pos()

            error = np.linalg.norm(kf-gt)
            print(key, kf, gt, error)
            self.plot_arr[4*j].append(error)
            self.plot_arr[4*j+1].append(np.abs(kf-gt)[0])
            self.plot_arr[4*j+2].append(np.abs(kf-gt)[1])
            self.plot_arr[4*j+3].append(np.abs(kf-gt)[2])

            self.axs[j].plot(self.time_arr, self.plot_arr[4*j], label=key+"_total_error")
            self.axs[j].plot(self.time_arr, self.plot_arr[4*j+1], label=key+"_x")
            self.axs[j].plot(self.time_arr, self.plot_arr[4*j+2], label=key+"_y")
            self.axs[j].plot(self.time_arr, self.plot_arr[4*j+3], label=key+"_z")


        """        

        for j, obstacle in enumerate(self.sim_gt_obstacles):
            for i in range(3):
                self.plot_arr[4*j+i].append(obstacle.get_pos()[i])
            self.plot_arr[4*j+3].append(obstacle.scaling/2)

        for i in range(4):
            self.plot_arr[8+i].append(obstacle_tracking.kf["small"].x[i])
            self.plot_arr[12+i].append(obstacle_tracking.kf["big"].x[i])

        

        for i in range(len(self.plot_arr)):
            if len(self.time_arr) == len(self.plot_arr[i]):
                self.axs[i%4].plot(self.time_arr, self.plot_arr[i], label=self.plot_title[i])

        """

        for ax in self.axs:
            ax.legend()

        plt.draw()
        plt.pause(0.001)



    def plot_trajectory(self, trajectory, trajectory_support_points, obstacle_tracking, nextTarget, currentEE):    
        plt.clf()

        for obstacle in obstacle_tracking:
            plt.plot(obstacle[0], obstacle[1], 'o', color='red')

        #for obstacle in self.sim_gt_obstacles:
        #    plt.plot(obstacle.get_pos()[0], obstacle.get_pos()[1], 'o', color='purple')

        if len(trajectory_support_points) > 0: plt.plot(trajectory_support_points[:, 0], trajectory_support_points[:, 1], color="grey")
        if len(trajectory) > 0: plt.plot(trajectory[:, 0], trajectory[:, 1], color="black")

        plt.plot(currentEE[0], currentEE[1], 'o', color='yellow')

        plt.plot(nextTarget[0], nextTarget[1], 'o', color='orange')


        plt.draw()
        plt.pause(0.001)


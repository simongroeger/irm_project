import time
import numpy as np
from simulation import Simulation, Camera
from obstacle_tracking import ObstacleTracking

import matplotlib.pyplot as plt


sim = Simulation(cam_pose=np.array([0.5, 0.0, 3.5]),
                 target_pose=np.array([0.5, 1e-20, 1.0]),
                 target_object="YcbBanana",
                 randomize=False)

robot = sim.get_robot()

obstacle_tracking = ObstacleTracking(sim.cam_matrices)

robot.print_joint_infos()

# rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM)
rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM, debug=True)
rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM, debug=True)

fig, axs = plt.subplots(1,4)

plot_arr = []
for i in range(17):
    plot_arr.append([])

print("start looping")
for time_step in range(10000):
    start = time.time()
    plot_arr[0].append(time_step)

    for j, obstacle in enumerate(sim.obstacles):
        print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)
        for i in range(3):
            plot_arr[1+4*j+i].append(obstacle.get_pos()[i])
        plot_arr[1+4*j+3].append(obstacle.scaling/2)


    rgb_fixed, depth_fixed = sim.get_renders(cam_type=Camera.FIXEDCAM)
    rgb_custom, depth_custom = sim.get_renders(cam_type=Camera.CUSTOMCAM)
    obstacle_tracking.step(rgb_fixed, depth_fixed, rgb_custom, depth_custom)


    for i in range(4):
        plot_arr[9+i].append(obstacle_tracking.kf["a"].x[i])
        plot_arr[13+i].append(obstacle_tracking.kf["b"].x[i])

    print("kf a", obstacle_tracking.kf["a"].x)
    print("kf b", obstacle_tracking.kf["b"].x)

    # robot.do_something()

    sim.step()

    end = time.time()
    print("step took", end - start, "s \n")

    for ax in axs:
        ax.clear()
    for i in range(1, len(plot_arr)):
        if len(plot_arr[0]) == len(plot_arr[i]):
            axs[(i-1)%4].plot(plot_arr[0], plot_arr[i], label=str(i))

    for ax in axs:
        ax.legend()
    plt.draw()
    plt.pause(0.01)

    a = 5

    


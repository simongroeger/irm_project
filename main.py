import time
import numpy as np
from simulation import Simulation, Camera
from grasp_sample_example import sample_grasps
from obstacle_tracking import ObstacleTracking
import matplotlib.pyplot as plt
from trajectory_planning import TrajectoryPlanning


sim = Simulation(cam_pose=np.array([0.5, 0.0, 3.5]),
                 target_pose=np.array([0.5, 1e-20, 1.0]),
                 target_object="YcbBanana",
                 randomize=False)

robot = sim.get_robot()

obstacle_tracking = ObstacleTracking(sim.cam_matrices)

trajectory_planning  = TrajectoryPlanning()

robot.print_joint_infos()

# rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM)
#rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM, debug=True)
# grasp, grasp_t, grasp_r = sample_grasps(sim)
# print(grasp, grasp_t, grasp_r)
#grasp_t = np.array([0.09507803, -0.65512755, 1.30783048])
#grasp_r = np.array([0.98625567, -0.15158182, -0.04823348, -0.04467924])
# grasp_r = np.array(
#     [
#         -0.17247438943737103,
#         0.9647538542107706,
#         -0.18215541132147892,
#         -0.07951095459099955,
#     ]
# )

state = "start"


#time.sleep(5)
#for _ in range(10000):
#    robot.do()
#    sim.step()


rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM, debug=True)
rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM, debug=True)

#fig, axs = plt.subplots(1,4)

plot_arr = []
for i in range(17):
    plot_arr.append([])

print("start looping")
for time_step in range(10000):
    start = time.time()
    #plot_arr[0].append(time_step)

    #for j, obstacle in enumerate(sim.obstacles):
        #print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)
        #for i in range(3):
        #    plot_arr[1+4*j+i].append(obstacle.get_pos()[i])
        #plot_arr[1+4*j+3].append(obstacle.scaling/2)


    rgb_fixed, depth_fixed = sim.get_renders(cam_type=Camera.FIXEDCAM)
    rgb_custom, depth_custom = sim.get_renders(cam_type=Camera.CUSTOMCAM)

    if time_step % (240//20) == 0:
        print("kf update")
        obstacle_tracking.step(rgb_fixed, depth_fixed, rgb_custom, depth_custom)


    #for i in range(4):
    #    plot_arr[9+i].append(obstacle_tracking.kf["a"].x[i])
    #    plot_arr[13+i].append(obstacle_tracking.kf["b"].x[i])

    #print("kf a", obstacle_tracking.kf["a"].x)
    #print("kf b", obstacle_tracking.kf["b"].x)

    trajectory, trajectory_support_points = trajectory_planning.plan([obstacle_tracking.kf["a"].x, obstacle_tracking.kf["b"].x])

    # robot do
    robot.do(trajectory)
    sim.step()

    end = time.time()
    #print("step took", end - start, "s \n")

    plt.clf()

    plt.plot(obstacle_tracking.kf["a"].x[0], obstacle_tracking.kf["a"].x[1], 'o', color='red')
    plt.plot(obstacle_tracking.kf["b"].x[0], obstacle_tracking.kf["b"].x[1], 'o', color='red')

    plt.plot(trajectory_support_points[:, 0], trajectory_support_points[:, 1], color="grey")
    if len(trajectory)  > 0: plt.plot(trajectory[:, 0], trajectory[:, 1], color="black")

    next_target = robot.getNextTarget(trajectory)
    current_ee = robot.ee_position()[0]
    plt.plot(current_ee[0], current_ee[1], 'o', color='yellow')
    plt.plot(next_target[0], next_target[1], 'o', color='orange')


    plt.draw()
    plt.pause(0.001)

    #for ax in axs:
    #    ax.clear()
    #for i in range(1, len(plot_arr)):
    #    if len(plot_arr[0]) == len(plot_arr[i]):
    #        axs[(i-1)%4].plot(plot_arr[0], plot_arr[i], label=str(i))

    #for ax in axs:
    #    ax.legend()
    #plt.draw()
    #plt.pause(0.01)

    #a = 5

    


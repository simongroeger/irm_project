import time
import numpy as np
from simulation import Simulation, Camera
from grasp_sample_example import sample_grasps
import pybullet as p


sim = Simulation(
    cam_pose=np.array([0.0, -0.65, 1.7]),
    target_pose=np.array([0, 0, -1]),
    target_object="YcbBanana",
    randomize=False,
)

robot = sim.get_robot()

robot.print_joint_infos()

# rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM)
# rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM, debug=True)
# grasp, grasp_t, grasp_r = sample_grasps(sim)
# print(grasp, grasp_t, grasp_r)
# grasp_t = np.array([0.09507803, -0.65512755, 1.30783048])
# grasp_r = np.array([0.98625567, -0.15158182, -0.04823348, -0.04467924])
# grasp_r = np.array(
#     [
#         -0.17247438943737103,
#         0.9647538542107706,
#         -0.18215541132147892,
#         -0.07951095459099955,
#     ]
# )

# time.sleep(5)
# for _ in range(10000):
#    robot.do()
#    sim.step()


# rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM, debug=True)
# rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM, debug=True)
# TODO: enable
# grasp = None
# while grasp is None:
#     grasp = sample_grasps(sim)
#     if len(grasp) == 0:
#         continue
#     else:
#         grasp_t = grasp[0]
#         grasp_r = grasp[1]
#         robot.set_grasp(grasp_r, grasp_t)
print("start looping")
for time_step in range(10000):
    start = time.time()

    if time_step % (240 // 20) == 0:
        rgb_fixed, depth_fixed = sim.get_renders(cam_type=Camera.FIXEDCAM)
        # rgb_custom, depth_custom = sim.get_renders(cam_type=Camera.CUSTOMCAM)
        print("kf update")
        robot.obstacle_tracking.step(rgb_fixed, depth_fixed)

    cmd = robot.do(sim)
    if cmd == "start":
        while True:
            grasp = sample_grasps(sim)
            if grasp is None:
                continue
            else:
                grasp_t = grasp[0] - np.array([0, 0, 0.035])
                grasp_r = grasp[1]
                robot.set_grasp(grasp_r, grasp_t)
                break
    sim.step()

    end = time.time()


# fig, axs = plt.subplots(1,4)

# plot_arr = []
# for i in range(17):
#    plot_arr.append([])

# plot_arr[0].append(time_step)

# for j, obstacle in enumerate(sim.obstacles):
# print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)
# for i in range(3):
#    plot_arr[1+4*j+i].append(obstacle.get_pos()[i])
# plot_arr[1+4*j+3].append(obstacle.scaling/2)


# for i in range(4):
#    plot_arr[9+i].append(obstacle_tracking.kf["a"].x[i])
#    plot_arr[13+i].append(obstacle_tracking.kf["b"].x[i])

# print("kf a", obstacle_tracking.kf["a"].x)
# print("kf b", obstacle_tracking.kf["b"].x)


# print("step took", end - start, "s \n")

# for ax in axs:
#    ax.clear()
# for i in range(1, len(plot_arr)):
#    if len(plot_arr[0]) == len(plot_arr[i]):
#        axs[(i-1)%4].plot(plot_arr[0], plot_arr[i], label=str(i))

# for ax in axs:
#    ax.legend()
# plt.draw()
# plt.pause(0.01)

# a = 5

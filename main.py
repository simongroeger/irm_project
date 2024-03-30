import time
import numpy as np
from simulation import Simulation, Camera
import pybullet as p
import vis

sim = Simulation(
    cam_pose=np.array([0.0, -0.65, 1.7]),
    target_pose=np.array([0, 0, -1]),
    target_object="YcbBanana",
    randomize=False,
)

robot = sim.get_robot()

robot.print_joint_infos()

vis.sim = sim

print("start looping")

for time_step in range(10000):
    start = time.time()

    if time_step % (240 // 40) == 0:
        #print("kf update")

        rgb_fixed, depth_fixed = sim.get_renders(cam_type=Camera.FIXEDCAM)
        robot.obstacle_tracking.step(rgb_fixed, depth_fixed, cam_type=Camera.FIXEDCAM)

        #for j, obstacle in enumerate(sim.obstacles):
        #    print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)
        #print("kf a", robot.obstacle_tracking.kf["a"].x)
        #print("kf b", robot.obstacle_tracking.kf["b"].x)

        #vis.plot_kf_error(time_step, robot.obstacle_tracking)

    cmd = robot.do(sim)
    
    sim.step()

    end = time.time()


import time
import numpy as np
from simulation import Simulation, Camera
from obstacle_tracking import ObstacleTracking

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

print("start looping")
for _ in range(10000):
    start = time.time()

    for obstacle in sim.obstacles:
        print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)


    rgb_fixed, depth_fixed = sim.get_renders(cam_type=Camera.FIXEDCAM)
    rgb_custom, depth_custom = sim.get_renders(cam_type=Camera.CUSTOMCAM)
    res = obstacle_tracking.step(rgb_fixed, depth_fixed, rgb_custom, depth_custom)

    print(res)

    # robot.do_something()

    end = time.time()
    print("step took", end - start, "s \n")
    sim.step()


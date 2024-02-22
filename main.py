import time
import numpy as np
from simulation import Simulation, Camera
from grasp_sample_example import sample_grasps

sim = Simulation(
    cam_pose=np.array([0.0, -1, 2.0]),
    target_pose=np.array([0, 0, 0]),
    target_object="YcbBanana",
    randomize=False,
)

robot = sim.get_robot()

robot.print_joint_infos()

# rgb, depth = sim.get_renders(cam_type=Camera.FIXEDCAM)
rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM, debug=True)
# grasp, grasp_t, grasp_r = sample_grasps(sim)
# print(grasp, grasp_t, grasp_r)
grasp_t = np.array([0.09507803, -0.65512755, 1.30783048])
grasp_r = np.array([0.98625567, -0.15158182, -0.04823348, -0.04467924])
# grasp_r = np.array(
#     [
#         -0.17247438943737103,
#         0.9647538542107706,
#         -0.18215541132147892,
#         -0.07951095459099955,
#     ]
# )
state = "start"

time.sleep(5)
for _ in range(10000):
    robot.do()
    sim.step()

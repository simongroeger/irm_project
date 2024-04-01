import time
import numpy as np
from simulation import Simulation, Camera
import pybullet as p
from vis import Vis
from project import Project


sim = Simulation(
    cam_pose=np.array([0.0, -0.65, 1.7]),
    target_pose=np.array([0, 0, -1]),
    target_object="YcbBanana",
    randomize=False,
)

robot = sim.get_robot()

robot.print_joint_infos()

vis = Vis(sim.obstacles)

project = Project(robot, sim.cam_matrices, sim.get_renders, sim.height, sim.width, vis)

print("start looping")

for time_step in range(10000):
    start = time.time()

    project.step(time_step)

    sim.step()

    end = time.time()

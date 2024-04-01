import numpy as np

from trajectory_planning import TrajectoryPlanning
from obstacle_tracking import ObstacleTracking
from grasp_sampler import GraspSampler
import vis
import pybullet as p
import sys
from robot import Robot


DOWN_GRIPPER = np.array(
    [
        0,
        1,
        0,
        0,
    ]
)
TARGET_GRIPPER = np.array(
    [0.9403202588231775, -0.1894998144336284, -0.06744521400733586, -0.2744790962602391]
)


class Project:
    def __init__(self, robot, cam_matrices, get_renders, height, width, vis) -> None:

        self.start_position = np.array([0.2, -0.2, 1.49])
        self.target_position = np.array([0.525, 0.675, 1.49])

        self.robot: Robot = robot
        self.vis = vis
        self.trajectory_planning = TrajectoryPlanning(
            self.start_position, self.target_position
        )
        self.grasp_sampler = GraspSampler(get_renders, cam_matrices, height, width)
        self.obstacle_tracking = ObstacleTracking(get_renders, cam_matrices)

        self.state =  "restart"
        self.gripper_state = "open"
        self.r_object = None
        self.gripper_t = None
        self.consecutive_fails = 0

        self.steps_in_state = 0

    def step(self, time_step):
        last_state = self.state
        currentEE = np.array(self.robot.ee_position()[0])
        # print(self.state, self.ee_position()[1], self.r_des)

        if (time_step % (240 // self.obstacle_tracking.measurement_hz) == 0):  # kalman filter only runs 40hz
            self.obstacle_tracking.step()
            #self.vis.plot_kf_error(time_step, self.obstacle_tracking)
        

        if self.gripper_state == "open":
            self.robot.open_gripper()
        else:
            self.robot.close_gripper()

        if self.state == "restart":
            if (
                self.robot.check_if_ee_reached(self.start_position)
                and self.robot.check_if_ee_is_stopped()
            ):
                self.robot.gripper_default_position()
                self.state = "start"
            else:
                self.robot.control(self.start_position, DOWN_GRIPPER, 1)

        elif self.state == "start":
            print("sampling grasps")
            while True:
                grasp = self.grasp_sampler.sample_grasps()
                if grasp is None:
                    continue
                else:
                    self.r_object = grasp[1]
                    self.gripper_t = grasp[0] - np.array([0, 0, 0.04])
                    print("got grasp:", self.gripper_t, self.r_object)
                    break
            self.state = "open_gripper"

        elif self.state == "open_gripper":
            if self.robot.check_if_gripper_open():
                self.state = "go_to_preposition_gripper"
            else:
                self.gripper_state = "open"

        elif self.state == "go_to_preposition_gripper":
            preposition = self.gripper_t + np.asarray(
                p.getMatrixFromQuaternion(self.r_object)
            ).reshape(3, 3) @ np.array([0, 0, -0.1])
            if self.robot.check_if_ee_reached(
                preposition
            ) and self.robot.check_if_ee_reached_orientation(self.r_object):
                self.state = "go_to_gripper"
            else:
                self.robot.control(preposition, self.r_object, 2)

        elif self.state == "go_to_gripper":
            if self.robot.check_if_ee_reached(self.gripper_t):
                self.state = "close_gripper"
            else:
                if 0.02 < self.robot.distance_to_target(self.gripper_t) < 0.06:
                    self.consecutive_fails += 1
                else:
                    self.consecutive_fails = 0
                if self.consecutive_fails > 100:
                    self.state = "restart"
                else:
                    self.robot.control(self.gripper_t, self.r_object, 1)

        elif self.state == "close_gripper":
            if self.robot.check_if_gripper_closed() and self.steps_in_state > 240/5:
                self.state = "lift_closed_gripper"
            else:
                self.gripper_state = "close"
                self.robot.control(self.gripper_t, self.r_object, 1)

        elif self.state == "lift_closed_gripper":
            if self.robot.check_if_gripper_is_empty():
                    self.state = "restart"
                    self.gripper_state = "open"
                    print("Gripper is empty")
            
            if self.robot.check_if_ee_reached(self.gripper_t + np.array([0, 0, 0.25])):
                self.state = "go_to_start"
            else:
                self.robot.control(self.gripper_t + np.array([0, 0, 0.25]), self.r_object, 1)

        elif self.state == "go_to_start":
            if self.robot.check_if_ee_reached(self.start_position):
                self.state = "go_to_target"
            else:
                self.vis.plot_trajectory(
                    [],
                    [],
                    self.obstacle_tracking.get_obstacles(),
                    self.start_position,
                    currentEE,
                )
                self.robot.control(self.start_position, DOWN_GRIPPER, 1)

        elif self.state == "go_to_target":
            if self.robot.check_if_ee_reached(self.target_position, neccessary_distance=0.05):
                self.state = "deliver"
            else:
                sucess, trajectory, trajectory_support_points = (
                    self.trajectory_planning.plan(
                        self.obstacle_tracking.get_obstacles()
                    )
                )
                current_target = self.trajectory_planning.getReferencePoint(
                    currentEE, trajectory
                )

                if not sucess:
                    self.state = "go_to_start"
                else:
                    self.vis.plot_trajectory(
                        trajectory,
                        trajectory_support_points,
                        self.obstacle_tracking.get_obstacles(),
                        current_target,
                        currentEE,
                    )
                    #for i in range(1, len(trajectory)):
                    #    p.addUserDebugLine(trajectory[i-1], trajectory[i], [0, 1, 0], lifeTime=2.0)
                    if self.robot.distance_to_target(self.target_position) < 0.4:
                        r_des = TARGET_GRIPPER
                    else:
                        r_des = DOWN_GRIPPER
                    self.robot.control(current_target, r_des, 4)

        elif self.state == "deliver":
            self.gripper_state = "open"
            if self.robot.check_if_gripper_open() and self.steps_in_state > 240/5:
                self.state = "done"
            else:
                self.robot.control(self.target_position, TARGET_GRIPPER, 2)

        elif self.state == "done":
            if (
                self.robot.check_if_ee_reached(self.start_position)
                and self.robot.check_if_ee_is_stopped()
            ):
                print("back at start, end simulation")
                #sys.exit()
            else:
                self.vis.plot_trajectory(
                    [],
                    [],
                    self.obstacle_tracking.get_obstacles(),
                    self.start_position,
                    currentEE,
                )
                self.robot.control(self.start_position, DOWN_GRIPPER, 1)

        if self.state != last_state:
            print(last_state, "->", self.state)
            self.steps_in_state = 0
        else:
            self.steps_in_state += 1
            

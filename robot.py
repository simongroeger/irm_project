import pybullet as p
import pybullet_robots
import numpy as np
from typing import Tuple
import sys


class Robot:
    """Robot Class.

    The class initializes a franka robot.

    Args:
        init_position: Initial position of the robot.
        orientation: Robot orientation in axis angle representation.
        table_scaling: Scaling parameter for the table.
    """

    def __init__(
        self,
        init_position: Tuple[float, float, float] = [0, 0, 0.62],
        orientation: Tuple[float, float, float] = [0, 0, 0],
        table_scaling: float = 2.0,
    ):
        # load robot
        self.pos = init_position
        self.axis_angle = orientation
        self.tscale = table_scaling
        self.state = "start"
        # self.target_position = np.array([0.0, 0.65, 1.24]) + np.array([0, 0, 0.5])
        self.target_position = np.array([0.6, 0.75, 1.24]) + np.array([0, 0, 0.5])
        self.gripper_t = np.array([0.09507803, -0.65512755, 1.30783048]) + np.array(
            [0, 0, -0.05]
        )
        if self.tscale != 1.0:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] * self.tscale]
        self.ori = p.getQuaternionFromEuler(self.axis_angle)

        self.arm_idx = [0, 1, 2, 3, 4, 5, 6]
        self.default_arm = [-1.8, 0.058, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.gripper_idx = [9, 10]

        self.ee_idx = 11

        self.id = p.loadURDF(
            "franka_panda/panda.urdf", self.pos, self.ori, useFixedBase=True
        )

        self.lower_limits, self.upper_limits = self.get_joint_limits()

        self.set_default_position()

        for j in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

    def set_default_position(self):
        for idx, pos in zip(self.arm_idx, self.default_arm):
            p.resetJointState(self.id, idx, pos)

    def get_joint_limits(self):
        lower = []
        upper = []
        for idx in self.arm_idx:
            joint_info = p.getJointInfo(self.id, idx)
            lower.append(joint_info[8])
            upper.append(joint_info[9])
        return lower, upper

    def print_joint_infos(self):
        num_joints = p.getNumJoints(self.id)
        print("number of joints are: {}".format(num_joints))
        for i in range(0, num_joints):
            print("Index: {}".format(p.getJointInfo(self.id, i)[0]))
            print("Name: {}".format(p.getJointInfo(self.id, i)[1]))
            print("Typ: {}".format(p.getJointInfo(self.id, i)[2]))

    def get_joint_positions(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return [state[0] for state in states]

    def get_all_joint_positions(self):
        idx = self.arm_idx + self.gripper_idx
        states = p.getJointStates(self.id, idx)
        return [state[0] for state in states]

    def ee_position(self):
        ee_info = p.getLinkState(self.id, self.ee_idx)
        ee_pos = ee_info[0]
        ee_ori = ee_info[1]
        return ee_pos, ee_ori

    def position_control(self, target_positions):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.arm_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )

    def get_joint_velocities(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return [state[1] for state in states]

    def get_joint_accelerations(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return [state[2] for state in states]

    def get_Jacobian(self):
        joint_pos = self.get_all_joint_positions()
        zero_vec = [0.0] * len(joint_pos)
        eeState = p.getLinkState(self.id, self.ee_idx)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        jac_t, jac_r = p.calculateJacobian(
            bodyUniqueId=self.id,
            linkIndex=self.ee_idx,
            localPosition=com_trn,
            objPositions=joint_pos,
            objVelocities=zero_vec,
            objAccelerations=zero_vec,
        )
        jac_t = np.asarray(jac_t)[:, : len(self.arm_idx)]
        jac_r = np.asarray(jac_r)[:, : len(self.arm_idx)]
        return jac_t, jac_r

    def JacobianPseudoinverseCtl(self, error_t, gain):
        jac_t, jac_r = self.get_Jacobian()
        jac_t_pinv = np.linalg.pinv(jac_t)
        v_ctl_t = np.dot(jac_t_pinv, error_t) * gain

        # error_r = np.asarray(p.getEulerFromQuaternion(error_r))
        # jac_r_pinv = np.linalg.pinv(jac_r)
        # v_ctl_r = np.dot(jac_r_pinv, error_r) * gain

        return v_ctl_t

    def Control(self, t_des, gain):
        t_curr, r_curr = self.ee_position()
        error_t = t_des - t_curr
        v_ctl_t = self.JacobianPseudoinverseCtl(error_t, gain)
        joint_positions = self.get_joint_positions()
        joint_velocities = v_ctl_t
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.arm_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
        )

    def check_if_ee_reached(self, t_des, neccessary_distrance=0.02):
        t_curr, _ = self.ee_position()
        return np.abs(np.linalg.norm(t_des - t_curr)) < neccessary_distrance

    def open_gripper(self):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.04, 0.04],
        )

    def check_if_gripper_open(self):
        states = p.getJointStates(self.id, self.gripper_idx)
        return states[0][0] > 0.03

    def close_gripper(self):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.01, 0.01],
        )

    def check_if_gripper_closed(self):
        states = p.getJointStates(self.id, self.gripper_idx)
        return states[0][0] < 0.02
    
    def getNextTarget(self, trajectory):
        current_ee = np.array(self.ee_position()[0])
        if len(trajectory) > 1:
            min_distance = 0
            min_index = -1
            for i, elem in enumerate(trajectory):
                current_distance = np.abs(np.linalg.norm(elem - current_ee))
                if min_index == -1 or current_distance < min_distance:
                    min_index = i
                    min_distance = current_distance

            target_index = min(len(trajectory)-1, min_index + 2)
            return trajectory[target_index]
        else:
            return current_ee

    def do(self, trajectory = []):
        state = self.state
        print(state)
        if state == "start":
            if self.check_if_gripper_open():
                self.state = "go_to_gripper"
            else:
                self.state = "open_gripper"
        elif state == "open_gripper":
            if self.check_if_gripper_open():
                self.state = "go_to_gripper"
            else:
                self.open_gripper()
                self.state = "open_gripper"
        elif state == "go_to_gripper":
            if self.check_if_ee_reached(self.gripper_t):
                self.state = "close_gripper"
            else:
                self.Control(self.gripper_t, 1)
                self.state = "go_to_gripper"
        elif state == "close_gripper":
            if self.check_if_gripper_closed():
                self.state = "go_to_target"
            else:
                self.close_gripper()
                self.Control(self.gripper_t, 1)
                self.state = "close_gripper"
        elif state == "go_to_target":
            if self.check_if_ee_reached(self.target_position, neccessary_distrance=0.05):
                self.state = "deliver"
            else:
                current_target = self.getNextTarget(trajectory)

                print("current", self.ee_position()[0], "next", current_target)
                
                self.Control(current_target, 1)
                self.state = "go_to_target"
        elif state == "deliver":
            if self.check_if_gripper_open():
                self.state = "done"
            else:
                self.Control(self.target_position, 1)
                self.open_gripper()
                self.state = "deliver"
        elif state == "done":
            self.Control(np.array(self.ee_position()[0]), 1)
            sys.exit()

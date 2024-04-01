import copy
import pybullet as p
import pybullet_robots
import numpy as np
from typing import Tuple
from pyquaternion import Quaternion


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

        if self.tscale != 1.0:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] * self.tscale]
        self.ori = p.getQuaternionFromEuler(self.axis_angle)

        self.arm_idx = [0, 1, 2, 3, 4, 5, 6]
        self.default_arm = [-1.8, 0.058, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.gripper_idx = [9, 10]

        self.ee_idx = 11
        self.consecutive_contacts = 0

        self.id = p.loadURDF(
            "franka_panda/panda.urdf", self.pos, self.ori, useFixedBase=True
        )

        self.lower_limits, self.upper_limits = self.get_joint_limits()

        self.set_default_position()

        for j in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

        p.setJointMotorControl2(
            self.id,
            self.gripper_idx[0],
            p.VELOCITY_CONTROL,
            force=0,
        )
        p.setJointMotorControl2(
            self.id,
            self.gripper_idx[1],
            p.VELOCITY_CONTROL,
            force=0,
        )

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

    def JacobianPseudoinverseCtl(self, u, gain_t, gain_r):
        jac_t, jac_r = self.get_Jacobian()
        jac = np.vstack((jac_t, jac_r))
        jac_pinv = np.linalg.pinv(jac)
        gain = np.array([gain_t, gain_t, gain_t, gain_r, gain_r, gain_r])
        v_ctl = np.dot(jac_pinv, gain * u)
        return v_ctl

    def control(self, t_des, r_des, gain):
        # control error
        u = np.zeros(6)

        # translational error
        t_curr, r_curr = self.ee_position()
        error_t = t_des - t_curr
        u[:3] = error_t

        # rotational error
        q_d = Quaternion(
            x=r_des[0],
            y=r_des[1],
            z=r_des[2],
            w=r_des[3],
        ).normalised
        q_e = Quaternion(x=r_curr[0], y=r_curr[1], z=r_curr[2], w=r_curr[3]).normalised
        q_r = q_d * q_e.conjugate
        u[3:] = q_r.elements[1:] * np.sign(q_r.elements[0])

        v_ctl_t = self.JacobianPseudoinverseCtl(u, gain, gain * 0.8)

        # limit joint velocities
        if np.max(v_ctl_t) > 0.4:
            v_ctl_t *= 0.4 / np.max(v_ctl_t)

        # joint_positions = self.get_joint_positions()
        joint_velocities = v_ctl_t
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.arm_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
        )

    def check_if_ee_reached(self, t_des, neccessary_distance=0.02):
        t_curr, _ = self.ee_position()
        return np.abs(np.linalg.norm(t_des - t_curr)) < neccessary_distance

    def check_if_ee_is_stopped(self):
        joint_velocities = self.get_joint_velocities()
        return np.linalg.norm(joint_velocities) < 0.01

    def check_if_ee_reached_orientation(self, r_des, neccessary_distance=0.02):
        _, r_curr = self.ee_position()
        q_d = Quaternion(x=r_des[0], y=r_des[1], z=r_des[2], w=r_des[3]).normalised
        q_e = Quaternion(x=r_curr[0], y=r_curr[1], z=r_curr[2], w=r_curr[3]).normalised
        q_r = q_d * q_e.conjugate
        # print("orientation error", np.abs(np.linalg.norm(q_r.elements[1:])))
        return np.abs(np.linalg.norm(q_r.elements[1:])) < neccessary_distance

    def distance_to_target(self, t_des):
        t_curr, _ = self.ee_position()
        # print("distance to target", np.linalg.norm(t_des - t_curr))
        return np.linalg.norm(t_des - t_curr)

    def open_gripper(self):
        # p.setJointMotorControlArray(
        #     self.id,
        #     jointIndices=self.gripper_idx,
        #     controlMode=p.POSITION_CONTROL,
        #     targetPositions=[0.04, 0.04],
        # )
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.gripper_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.5, 0.5],
            forces=[1, 1],
        )

    def check_if_gripper_open(self):
        states = p.getJointStates(self.id, self.gripper_idx)
        #print("gripper state", states[0][0])
        return states[0][0] > 0.04

    def close_gripper(self):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.gripper_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[-0.1, -0.1],
            forces=[10, 10],
        )

    def check_if_gripper_closed(self):
        if self.check_if_gripper_is_empty():
            return True
        contact1 = p.getContactPoints(bodyA=self.id, linkIndexA=self.gripper_idx[0])
        contact2 = p.getContactPoints(bodyA=self.id, linkIndexA=self.gripper_idx[1])

        force1 = np.mean([contact[9] for contact in contact1])
        force2 = np.mean([contact[9] for contact in contact2])
        # print("force", force1, force2)
        if force1 + force2 > 2:
            self.consecutive_contacts += 1
        else:
            self.consecutive_contacts = 0

        return self.consecutive_contacts > 120

    def check_if_gripper_is_empty(self):
        states = p.getJointStates(self.id, self.gripper_idx)
        return states[0][0] < 0.0001

    def gripper_default_position(self):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=[self.arm_idx[-1]],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.0],
        )

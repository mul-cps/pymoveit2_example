#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
from typing import Optional

from pymoveit2 import MoveIt2
from pymoveit2.robot_description import RobotDescription

from get_parameter import node_get_parameter_str


class RobotMove(Node):
    def __init__(self):
        super().__init__("move_node")
        self._callback_group = ReentrantCallbackGroup()
        self._description: Optional[RobotDescription] = None
        self.moveit2: Optional[MoveIt2] = None
        self._group_name: Optional[str] = None

    # ----------------------------------------------------------
    def setup(self, group_name: Optional[str] = None):
        # get URDF and SRDF
        urdf_xml = node_get_parameter_str(
            node=self,
            node_name="/robot_state_publisher",
            parameter_name="robot_description",
        )
        srdf_xml = node_get_parameter_str(
            node=self,
            node_name="/move_group",
            parameter_name="robot_description_semantic",
        )

        self._description = RobotDescription(urdf_xml = urdf_xml, srdf_xml = srdf_xml)

        self.moveit2 = MoveIt2(
            node=self,
            callback_group=self._callback_group,
            **self._description.moveit2_kwargs(group_name),
        )

        # Resolve the actual group name so later lookups (named states,
        # base link, etc.) don't fall back to a None key.
        self._group_name = group_name or next(iter(self._description.groups))

        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        self._log_description()

        self.get_logger().info(
            f"MoveIt2 controller ready (group '{self._group_name}')."
        )

    # ----------------------------------------------------------
    def _log_description(self):
        """Print the groups, links and named states parsed from the robot description."""
        for name, group in self._description.groups.items():
            self.get_logger().info(
                f"group '{name}': {group.base_link_name} -> {group.end_effector_name}"
            )
            self.get_logger().info(f"  joints: {group.joint_names}")
            for state_name, state_positions in self.named_states(name).items():
                self.get_logger().info(f"  state '{state_name}': {state_positions}")

    # ----------------------------------------------------------
    def named_states(self, group_name: Optional[str] = None) -> dict:
        """Return {state_name: joint_positions} for the given (or active) group."""
        group_name = group_name or self._group_name
        return self._description._states.get(group_name, {})

    def joint_state(self, state_name: str, group_name: Optional[str] = None):
        """Look up a named joint configuration, ordered to match group joint_names."""
        group_name = group_name or self._group_name
        states = self.named_states(group_name)
        if state_name not in states:
            raise KeyError(
                f"Named state '{state_name}' not found. "
                f"Available: {list(states.keys())}"
            )
        state = states[state_name]
        joint_names = self._description.groups[group_name].joint_names
        # Keep the group's joint order; default any missing joint to 0.0.
        return [state.get(j, 0.0) for j in joint_names]

    # ----------------------------------------------------------
    def move_to_state(self, state_name: str):
        """Plan and execute a joint-space goal referenced by its SRDF name."""
        self.move_to_joint_position(self.joint_state(state_name))

    def move_to_joint_position(self, joint_position):
        """Plan and execute a joint-space goal (radians)."""
        self.get_logger().info(f"Moving to joint goal: {joint_position}")
        self.moveit2.move_to_configuration(joint_position)
        self.moveit2.wait_until_executed()
        self.get_logger().info("Joint goal reached.")

    # ----------------------------------------------------------
    def move_to_pose(self, position, quat_xyzw):
        """
        Plan and execute a Cartesian pose goal.
        position   : [x, y, z] in meters
        quat_xyzw  : [x, y, z, w] orientation quaternion
        """
        self.get_logger().info(
            f"Moving to pose: pos={position}, quat={quat_xyzw}"
        )
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
        )
        self.moveit2.wait_until_executed()
        self.get_logger().info("Pose goal reached.")


def main():
    rclpy.init()
    node = RobotMove()

    # Spin the node in a separate thread so service/action calls resolve
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node.setup()

    # Give everything a moment to connect
    node.create_rate(1.0).sleep()

    try:
        # ---- 1. Move to a named configuration from the description ----
        node.move_to_state("ready")

        # $ ros2 run tf2_ros tf2_echo base_link tool0
        # - Translation: [-0.131, 0.205, 0.285]
        # - Rotation: in Quaternion (xyzw) [1.000, -0.000, 0.000, 0.001]
        # - Rotation: in RPY (radian) [3.141, -0.000, -0.000]
        # - Rotation: in RPY (degree) [179.938, -0.017, -0.021]
        # - Matrix:
        #   1.000 -0.000  0.000 -0.131
        #  -0.000 -1.000 -0.001  0.205
        #   0.000  0.001 -1.000  0.285
        #   0.000  0.000  0.000  1.000

        # ---- 2. Trace a four-corner diamond, beginning at the ready/bottom pose ----
        # Keep the existing Cartesian target as the bottom vertex. The diamond
        # is traced in the XY plane, then the end-effector returns to the bottom.
        bottom = [0.0, 0.205, 0.285]
        diamond_points = [
            bottom,
            [0.05, 0.205, 0.335],   # right
            [0.0, 0.205, 0.385],    # top
            [-0.05, 0.205, 0.335],  # left
            bottom,                 # close the diamond
        ]
        for point in diamond_points:
            node.move_to_pose(
                position=point,
                quat_xyzw=[1.0, 0.0, 0.0, 0.0],  # pointing down
            )

        # ---- 3. Return to the named 'ready' state ----
        node.move_to_state("ready")

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        rclpy.shutdown()
        spin_thread.join()


if __name__ == "__main__":
    main()

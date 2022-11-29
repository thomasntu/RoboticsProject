#!/usr/bin/env python
import math as m
import sys
from datetime import datetime
from turtle import shape
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from cv2 import rotate
from rclpy.node import Node

sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from sensor_msgs.msg import Image
from tm_msgs.msg import *
from tm_msgs.srv import *

from . import shapes
from .shapes import T, img2Camera


class ImageSub(Node):
    def __init__(self, nodeName):
        super().__init__(nodeName)
        self.subscription = self.create_subscription(Image,
        'techman_image', self.image_callback, 10)
        self.subscription

    def image_callback(self, data):
        self.get_logger().info('Received image')

        # TODO (write your code here)
        img = np.array(data.data).reshape(data.height, data.width, 3)

        cv2.imwrite(f'TEST_IMG{datetime.now()}.png', img)

        objs, _ = shapes.detect(img)
        self.get_logger().info(str(objs))

        target_z = 100

        # TODO: Tune robot arm orientation (rz)
        for object in objs[:1]:
            cx, cy, phi = object
            x, y, z = img2Camera((cx - 640, cy - 480))
            print(f"{x}, {y}, {z}")

            self.get_logger().info(f"Camera: {x}, {y}, {z}")

            x, y, z, _ = T @ np.array([x, y + 85, z, 1])
            angle = (135 - 90 - m.degrees(phi)) % 360

            self.get_logger().info(f"Target: {x}, {y}, {0}")
            self.get_logger().info(f"Angle: {angle}")

            frame = f"{x:.0f}, {y:.0f}, 200, -180.00, 0.0, {angle:.2f}"
            script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"

            send_script(script_ptp)

            set_io(0.0)

            frame = f"{x:.0f}, {y:.0f}, 100, -180.00, 0.0, {angle:.2f}"
            script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"

            send_script(script_ptp)

            set_io(1.0)

            # target = f"350, 350, {target_z:.0f}, -180.00, 0.0, 135"
            # target_ptp = "PTP(\"CPP\"," + target + ",100,200,0,false)"

            # target_z += 50

            # send_script(target_ptp)
            # set_io(0.0)

            # target = f"350, 350, {target_z + 100:.0f}, -180.00, 0.0, 135"
            # target_ptp = "PTP(\"CPP\"," + target + ",100,200,0,false)"

            # send_script(target_ptp)



def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

def set_io(state):
    gripper_node = rclpy.create_node('gripper')
    gripper_cli = gripper_node.create_client(SetIO, 'set_io')

    while not gripper_cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not availabe, waiting again...')

    io_cmd = SetIO.Request()
    io_cmd.module = 1
    io_cmd.type = 1
    io_cmd.pin = 0
    io_cmd.state = state
    gripper_cli.call_async(io_cmd)
    gripper_node.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSub('image_sub')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

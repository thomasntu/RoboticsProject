#!/usr/bin/env python
from datetime import datetime
from turtle import shape

from cv2 import rotate
import rclpy
import numpy as np
from typing import List, Tuple

import math as m
from rclpy.node import Node

import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *

from sensor_msgs.msg import Image
from . import shapes
import cv2

# from scipy.spatial.transfrom import Rotation as R


camera_matrix = np.array([[1.37792826e+03, 0.00000000e+00, 6.59752738e+02],
                          [0.00000000e+00, 1.37632357e+03, 5.44888633e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
x_arm = 350
y_arm = 350
z_arm = 730
phi = 135
camera_coord = np.array([0, -85, 0])

def rotate2D(phi: float) -> np.ndarray:
    phi = m.radians(phi)
    return np.array([[m.cos(phi), -m.sin(phi), 0],
                     [m.sin(phi), m.cos(phi), 0],
                     [0, 0, 1]])

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
        objs: List = shapes.detect(img)
        self.get_logger().info(str(objs))


        for object in objs:
            x, y, a = object
            image_coord = np.array([x, y, 1])
            # cam_rot = R.from_euler('xyz', [-180, 0, 135], degrees=True)
            # cam_trans = np.array([x_arm, y_arm, z_arm])
            # f_camera_matrix = camera_matrix * np.concatenate(cam_rot, cam_trans.T, axis=1)
            world_coord = (np.linalg.inv(camera_matrix) @ image_coord.T) * z_arm
            cam_trans = rotate2D(-phi) @ camera_coord
            rotated_point = rotate2D(-phi) @ world_coord
            final_coord = rotated_point + cam_trans + np.array([x_arm, y_arm, 0])
            targetP1 = f"{final_coord[0]}, {final_coord[1]}, 200, -180.00, 0.0, 135.00"
            # script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
            # send_script(script1)
            self.get_logger().info(f'INV: {np.linalg.inv(camera_matrix)}')
            self.get_logger().info(f'IMAGE: {image_coord}')
            self.get_logger().info(f'WORLD: {world_coord}')
            self.get_logger().info(f'FINAL: {final_coord}')
            


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

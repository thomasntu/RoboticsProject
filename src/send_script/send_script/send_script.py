#!/usr/bin/env python

import time
from typing import List
import rclpy

import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *

import cv2
from . import shapes
import numpy as np
import math as m
from os.path import exists
from os import remove

# arm client
def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

# gripper client
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

def loop():
    start = time.time()

    while True:

        filename = f'images/IMG.png'

        if exists(filename):
            time.sleep(1)
            image = cv2.imread(filename)

            remove(filename)

            objs: List = shapes.detect(image)
            print(f"OBJECTS:{str(objs)}")

            target_z = 100

            # TODO: Tune robot arm orientation (rz)
            for object in objs:
                cx, cy, phi = object
                x, y = shapes.img2world((cx, cy))
                angle = (135 - 90 - m.degrees(phi)) % 360

                frame = f"{x:.0f}, {y:.0f}, 200, -180.00, 0.0, {angle:.2f}"
                script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"

                send_script(script_ptp)

                set_io(0.0)

                frame = f"{x:.0f}, {y:.0f}, 100, -180.00, 0.0, {angle:.2f}"
                script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"

                send_script(script_ptp)

                set_io(1.0)

                target = f"350, 350, {target_z:.0f}, -180.00, 0.0, 135"
                target_ptp = "PTP(\"CPP\"," + target + ",100,200,0,false)"

                target_z += 50

                send_script(target_ptp)
                set_io(0.0)

                target = f"350, 350, {target_z + 100:.0f}, -180.00, 0.0, 135"
                target_ptp = "PTP(\"CPP\"," + target + ",100,200,0,false)"

                send_script(target_ptp)

        elif time.time() - start > 1000:
            break

def main(args=None):

    rclpy.init(args=args)

    
    #--- move command by joint angle ---#
    # script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'

    #--- move command by end effector's pose (x,y,z,a,b,c) ---#
    # targetP1 = "398.97, -122.27, 748.26, -179.62, 0.25, 90.12"s

    # Initial camera position for taking image (Please do not change the values)
    # For right arm: targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    # For left  arm: targetP1 = "350.00, 350, 730, -180.00, 0.0, 135.00"
    # set_io(0.0)

    targetP1 = "350, 350, 730, -180.00, 0.0, 135.00"
    script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
    send_script(script1)

    send_script("Vision_DoJob(job1)")

    set_io(0.0)

    loop()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
    


    

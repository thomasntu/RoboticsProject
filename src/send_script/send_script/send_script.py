#!/usr/bin/env python

import math as m
import time
from os import remove
from os.path import exists
from typing import List

import cv2
import numpy as np
import rclpy

from . import shapes
from .controller import send_script, set_io


def alpha_blend(a, b, alpha):
    return (np.array(a) * alpha + np.array(b) * (1 - alpha)).tolist()


def loop2():
    """
    Copy image
    """
    start = time.time()
    filename = f'images/IMG.png'

    while True:
        if exists(filename):
            time.sleep(1)

            image = cv2.imread(filename)
            remove(filename)

            # Go to a point close to the table while computing the path
            go_to(350, 350, 300)

            path_nodes, jump_nodes = shapes.calculate_path(image)

            prev_node = path_nodes[0]
            # Go to the first node
            go_to_point(prev_node, z=300)
            go_to_point(prev_node)

            for path_node in path_nodes:
                if prev_node in jump_nodes:
                    jump(prev_node, path_node)
                else:
                    go_to_point(path_node)
                prev_node = path_nodes

            # return to initial position
            go_to(350, 350, 730)

        elif time.time() - start > 120:
            break


def jump(p1, p2):
    go_to_point(p1, z=300)
    go_to_point(p2, z=300)
    go_to_point(p2)


def go_to_point(p, z=200):
    go_to(p[0], p[1], z)


def go_to(x, y, z):
    target_p1 = f"{x}, {y}, {z}, -180.00, 0.0, 135.00"
    script1 = "PTP(\"CPP\"," + target_p1 + ",100,200,0,false)"
    send_script(script1)


def take_picture():
    send_script("Vision_DoJob(job1)")


def main(args=None):
    rclpy.init(args=args)

    # Take a picture
    go_to(350, 350, 730)
    take_picture()

    # Enter into the main loop
    loop2()

    # Shutdown
    rclpy.shutdown()


if __name__ == '__main__':
    main()

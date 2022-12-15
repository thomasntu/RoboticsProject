#!/usr/bin/env python

import time
from os import remove
from os.path import exists

import cv2
import numpy as np
import rclpy
from tqdm import tqdm

from . import shapes
from .controller import send_script


def alpha_blend(a, b, alpha):
    return (np.array(a) * alpha + np.array(b) * (1 - alpha)).tolist()


def jump(p1, p2):
    go_to_point(p1, z=240)
    go_to_point(p2, z=240)
    go_to_point(p2)

# draw at 230 for pilot mine
# draw at 219 for thin mine
def go_to_point(p, z=219):
    go_to(p[0], p[1], z)


def go_to(x:float, y:float, z:float, a: float = -180.0, b: float = 0.0, c: float = 135.0):
    target_p1 = f"{x:.2f}, {y:.2f}, {z:.2f}, {a:.2f}, {b:.2f}, {c:.2f}"
    script1 = "PTP(\"CPP\"," + target_p1 + ",100,200,0,false)"
    send_script(script1)


def take_picture():
    send_script("Vision_DoJob(job2)")


def copy_image_loop():
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

            # TODO: Modify return type in calculate_path()
            # such that `if prev_node in jump_nodes` is valid
            path_nodes = np.array(path_nodes)
            jump_nodes = np.array(jump_nodes)

            prev_node = path_nodes[0]

            # Go to the first node
            go_to_point(prev_node, z=300)
            go_to_point(prev_node)

            for path_node in tqdm(path_nodes):
                if prev_node in jump_nodes:
                    jump(prev_node, path_node)
                else:
                    go_to_point(path_node)
                prev_node = path_node

            # return to initial position
            go_to(350, 350, 730)

            break

        elif time.time() - start > 120:
            break


def draw_face_loop():
    """
    Draw Face
    """

    # Take a picture of the face
    go_to(350, 350, 600, 90, 0, 45)
    take_picture()

    start = time.time()
    filename = f'images/IMG.png'

    images = []

    while True:
        if exists(filename):
            time.sleep(1)

            image = cv2.imread(filename)
            remove(filename)

            images.append(image)

        if len(images) == 2:

            # cv2.imshow('0', images[0])
            # cv2.imshow('1', images[1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Go to a point close to the table while computing the path
            go_to(350, 350, 300)

            # First image is canvas, second image is face
            path_nodes, jump_nodes = shapes.calculate_path_for_face(images[1], images[0])

            # TODO: Modify return type in calculate_path()
            # such that `if prev_node in jump_nodes` is valid
            path_nodes = np.array(path_nodes)
            jump_nodes = np.array(jump_nodes)

            prev_node = path_nodes[0]

            # Go to the first node
            go_to_point(prev_node, z=300)
            go_to_point(prev_node)

            for path_node in tqdm(path_nodes):
                if prev_node in jump_nodes:
                    jump(prev_node, path_node)
                else:
                    go_to_point(path_node)
                prev_node = path_node

            # return to initial position
            go_to(350, 350, 730)

            break

        elif time.time() - start > 120:
            break


def main(args=None):
    rclpy.init(args=args)

    # Take a picture
    go_to(275, 425, 730)
    take_picture()

    go_to(425, 275, 730)
    take_picture()

    # Enter into the main loop
    # draw_face_loop()
    # copy_image_loop()

    # Shutdown
    rclpy.shutdown()


if __name__ == '__main__':
    main()

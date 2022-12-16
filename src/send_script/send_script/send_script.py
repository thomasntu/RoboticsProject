#!/usr/bin/env python

import time
from os import remove
from os.path import exists

import cv2
import numpy as np
import rclpy
from tqdm import tqdm

from . import clean, shapes
from .controller import send_script


def alpha_blend(a, b, alpha):
    return (np.array(a) * alpha + np.array(b) * (1 - alpha)).tolist()


def jump(p1, p2):
    go_to_point(p1, z=245)
    go_to_point(p2, z=245)
    go_to_point(p2)


# draw at 230 for pilot mine
# draw at 219 for thin mine
# draw at 238 for black marker pan
# draw at 234 for damaged marker pan
def go_to_point(p, z=238):
    go_to(p[0], p[1], z)


def go_to(x: float, y: float, z: float, a: float = -180.0, b: float = 0.0, c: float = 135.0):
    target_p1 = f"{x:.2f}, {y:.2f}, {z:.2f}, {a:.2f}, {b:.2f}, {c:.2f}"
    script1 = "PTP(\"CPP\"," + target_p1 + ",100,200,0,false)"
    send_script(script1)


def take_picture(job=2):
    send_script(f"Vision_DoJob(job{job})")


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

            path_nodes, jump_nodes = clean.copy_image(image)

            draw(path_nodes, jump_nodes)

            break

        elif time.time() - start > 120:
            break


def draw_face_loop():
    """
    Draw Face
    """

    # Take a picture of the face
    go_to(450, 250, 650, 90, 0, 45)
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

            # Go to a point close to the table while computing the path
            go_to(350, 350, 300)

            # First image is canvas, second image is face
            path_nodes, jump_nodes = clean.draw_face(images[1], images[0])

            draw(path_nodes, jump_nodes)

            break

        elif time.time() - start > 120:
            break


def draw(path_nodes, jump_nodes):
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


def grab_3d_loop():
    start = time.time()
    filename = f'images/IMG.png'

    images = []

    idx = 0

    while True:
        if exists(filename):
            time.sleep(1)

            image = cv2.imread(filename)
            remove(filename)

            if idx == 0:
                # Take a picture of the face
                go_to(300, 300, 730)
                take_picture()
                idx += 1
                break
            elif idx == 1:
                images.append(image)
                go_to(400, 400, 730)
                take_picture()
            else:
                images.append(image)

        if len(images) == 2:

            objs1 = shapes.detect(images[0])
            objs2 = shapes.detect(images[1])

            coords = shapes.stereo2world(objs1[0][:2], objs2[0][:2])

            go_to(coords[0], coords[1], coords[2])

            break

        elif time.time() - start > 120:
            break


def main(args=None):
    rclpy.init(args=args)

    filename = f'images/IMG.png'
    if exists(filename):
        time.sleep(1)
        remove(filename)

    # Take a picture
    go_to(350, 350, 730)
    take_picture()
    # set_io(1.0)
    # go_to(326.5, 400, 100, b=0, c=0)

    # Enter into the main loop
    # draw_face_loop()
    # copy_image_loop()
    grab_3d_loop()

    # Shutdown
    rclpy.shutdown()


if __name__ == '__main__':
    main()

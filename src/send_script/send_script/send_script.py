#!/usr/bin/env python

import math as m
import time
from os import remove
from os.path import exists
from typing import List

import cv2
import numpy as np
import rclpy

from . import detector, shapes
from .controller import send_script, set_io


def alpha_blend(a, b, alpha):
    return (np.array(a) * alpha + np.array(b) * (1 - alpha)).tolist()


def loop():
    """
    Stack blocks
    """
    start = time.time()
    filename = f'images/IMG.png'

    while True:
        if exists(filename):
            # An image was recorded
            time.sleep(1)

            image = cv2.imread(filename)

            # Delete the image after reading
            remove(filename)

            # Detect centroids and principle angles
            objs: List = shapes.detect(image)
            print(f"OBJECTS:{str(objs)}")

            # Target height for stacking
            target_z = 100

            for _object in objs:
                cx, cy, phi = _object

                # Translate centroid into world coordinates
                x, y = shapes.img2world((cx, cy))

                # Add centroid angle to initial camera angle
                angle = (135 - 90 - m.degrees(phi)) % 360

                # Go to a point 50mm above the object and open the gripper
                frame = f"{x:.1f}, {y:.1f}, 150, -180.00, 0.0, {angle:.2f}"
                script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"
                send_script(script_ptp)
                set_io(0.0)

                # Go down to the object and close the gripper
                frame = f"{x:.1f}, {y:.1f}, 100, -180.00, 0.0, {angle:.2f}"
                script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"
                send_script(script_ptp)
                set_io(1.0)

                # Go to a point 100mm above the object location
                frame = f"{x:.1f}, {y:.1f}, 200, -180.00, 0.0, {angle:.2f}"
                script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"
                send_script(script_ptp)

                # Go to the stacking location, update the stack height and open the gripper
                target = f"350, 350, {target_z:.0f}, -180.00, 0.0, 135"
                target_ptp = "PTP(\"CPP\"," + target + ",100,200,0,false)"
                target_z += 30
                send_script(target_ptp)
                set_io(0.0)

                # Go to a point 130mm above the stacking location
                target = f"350, 350, {target_z + 100:.0f}, -180.00, 0.0, 135"
                target_ptp = "PTP(\"CPP\"," + target + ",100,200,0,false)"
                send_script(target_ptp)

        elif time.time() - start > 120:
            # Exit loop after 2 minutes
            break


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

            contours = detector.find_contours(image)
            rects = list(filter(lambda x: bool(x), [detector.find_contour_with_n_corner(c) for c in contours]))
            print(f"Rectangles: {rects}")

            for rect in rects[:1]:
                tl, tr, br, bl = rect
                center = detector.line_intersection((tl, br), (tr, bl))

                # Draw rectangles
                # for corner in rect:
                #     # center = detector.line_intersection((tl, br), (tr, bl))
                #     x, y = shapes.img2world(alpha_blend(center, corner, 0.75))       # top_l in world coordinate
                #     # rect = [shapes.img2world(x, y) for x, y in rect]
                #     # x, y = shapes.img2world(x, y)

                #     # TODO:
                #     # - implement class for generating command.
                #     # - send_script() is non-blocking
                #     frame = f"{x:.1f}, {y:.1f}, 250, -180.00, 0.0, 135.00"
                #     script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"

                #     send_script(script_ptp)

                #     # TODO: rotate and face to next node
                #     frame = f"{x:.1f}, {y:.1f}, 250, -180.00, 0.0, 135.00"
                #     script_ptp = "PTP(\"CPP\"," + frame + ",100,300,0,false)"

                #     send_script(script_ptp)

                # Draw circles:
                # (We treat the stroke as combination of line segments)
                # TMRobot constructs circle by 3 points:
                #  - Current position -> center
                #  - Point it passed by
                #  - End point
                # radius = 300
                # pb = (center[0] + radius * m.cos(0), center[1] + radius * m.sin(0))
                # pe = (center[0] + radius * m.cos(m.radians(270)), center[1] + radius * m.sin(m.radians(270)))
                # sx, sy = shapes.img2world(pb)       # top_l in world coordinate
                # ex, ey = shapes.img2world(pe)

                # x, y = shapes.img2world(center)
                # frame = f"{x:.1f}, {y:.1f}, 250, -180.00, 0.0, 135.00"

                # script_ptp = f"PTP(\"CPP\",{frame},100,300,0,false)"
                # send_script(script_ptp)

                # pb = f"{sx:.1f}, {sy:.1f}, 250, -180.00, 0.0, 135.00"
                # pe = f"{ex:.1f}, {ey:.1f}, 250, -180.00, 0.0, 135.00"

                # script_circle = f"Circle(\"CPP\",{pb},{pe},100,200,50,270,false)"
                # send_script(script_circle)

        elif time.time() - start > 120:
            break


def main(args=None):
    rclpy.init(args=args)

    # --- move command by joint angle ---#
    # script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'

    # --- move command by end effector's pose (x,y,z,a,b,c) ---#
    # targetP1 = "398.97, -122.27, 748.26, -179.62, 0.25, 90.12"s

    # Initial camera position for taking image (Please do not change the values)
    # For right arm: targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    # For left  arm: targetP1 = "350.00, 350, 730, -180.00, 0.0, 135.00"
    # set_io(0.0)

    # Take a picture
    target_p1 = "350, 350, 730, -180.00, 0.0, 135.00"
    script1 = "PTP(\"CPP\"," + target_p1 + ",100,200,0,false)"
    send_script(script1)
    send_script("Vision_DoJob(job1)")

    # Enter into the main loop
    loop2()

    # Shutdown
    rclpy.shutdown()


if __name__ == '__main__':
    main()

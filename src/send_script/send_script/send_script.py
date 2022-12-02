#!/usr/bin/env python

import time
from typing import List
import rclpy

from time import sleep
import cv2
from . import detector, shapes
from .controller import send_script, set_io
import numpy as np
import math as m
from os.path import exists
from os import remove

def alpha_blend(a, b, alpha):
    return (np.array(a) * alpha + np.array(b) * (1 - alpha)).tolist()

def loop():
    start = time.time()

    while True:

        filename = f'images/IMG.png'

        if exists(filename):
            time.sleep(1)

            image = cv2.imread(filename)
            remove(filename)

            contours = detector.find_rect(image)
            rects = list(filter(lambda x: bool(x), [detector.find_corner(c) for c in contours]))
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
 
    #--- move command by joint angle ---#
    # script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'

    #--- move command by end effector's pose (x,y,z,a,b,c) ---#
    # targetP1 = "398.97, -122.27, 748.26, -179.62, 0.25, 90.12"s

    # Initial camera position for taking image (Please do not change the values)
    # For right arm: targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    # For left  arm: targetP1 = "350.00, 350, 730, -180.00, 0.0, 135.00"
    set_io(1.0)

    targetP1 = "350, 350, 730, -180.00, 0.0, 135.00"
    script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
    send_script(script1)

    send_script("Vision_DoJob(job1)")
    # sleep(1)
    # set_io(1.0)

    loop()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
    


    

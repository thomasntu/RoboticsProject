import argparse
import math as m
from operator import inv
import os
from math import atan2, degrees, tan
from typing import List, Tuple

import cv2
from cv2 import blur
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

# Type definition
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]    # (x, y, z)
Frame2D = Tuple[float, float, float]    # (x, y, phi)
Frame3D = Tuple

# Measured at:
# "     x,   y,   z,      rx,  ry,     rz"
# "350.00, 350, 730, -180.00, 0.0, 135.00"
intrinsic = np.array([
    [1.37792826e+03, 0.00000000e+00, 6.59752738e+02],
    [0.00000000e+00, 1.37632357e+03, 5.44888633e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
distortion = np.array([[
    1.54761726e-01, -4.84702069e-01,  3.24862920e-04,  1.57942165e-02, -3.63065079e-01
]])
undistorted_intrinsic = np.array([
    [1.30196814e+03, 0.00000000e+00, 6.95057417e+02],
    [0.00000000e+00, 1.28375269e+03, 5.58235465e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
# roi = (60, 58, 1204, 880)
f = 1376.0

arm_position = (350, 350, 730)
arm_orientation = (-180, 0, 135)
rot = R.from_euler('xyz', arm_orientation, degrees=True).as_matrix()
offset = (0, -85, 0)

t = (rot @ -np.array([arm_position]).T) + np.array([offset]).T
extrinsic = np.concatenate((rot, t), axis=1)

# Homogeneous transformation from camera to base
T = np.concatenate(
    (np.concatenate((rot, [[350], [350], [730]]), axis=1), [[0, 0, 0, 1]]), axis=0
)

# TODO: Maybe, we can split the modules as detector and controller.

def rotate2D(phi: float) -> np.ndarray:
    phi = m.radians(phi)

    return np.array([
        [m.cos(phi), -m.sin(phi), 0],
        [m.sin(phi), m.cos(phi), 0],
        [0, 0, 1]]
    )


def principleLine(w: int, frame: Frame2D) -> Point2D:
    cx, cy, phi = frame

    # y - cY = k(x - cX)
    k = tan(phi)
    point1 = (0, -int(k * cx) + cy)
    point2 = (w, int(k * (w - cx) + cy))

    return (point1, point2)


def detect(img: np.ndarray) -> List[Frame2D]:
    """Detect objects based on binary thresholding method.
    Return the coordinate of central points (pixel) and principle angle (radian) in image frame.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    objects = []
    for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        moment = cv2.moments(c)

        # Drop contour if the area is too small.
        if moment["m00"] < 600:
            continue

        cX = moment["m10"] / moment["m00"]
        cY = moment["m01"] / moment["m00"]

        # Principle angle 'phi' and slope 'k' = 'tan(phi)'
        phi = atan2(2 * moment["mu11"], (moment["mu20"] - moment["mu02"])) / 2
        objects.append((cX, cY, phi))

    return objects

# Function for debugging.
# To learn how the coordinate in 3D space maps to 2D space.
def camera2img(points: Point3D) -> np.ndarray:
    x, y, z = points
    p = [x, y, z, 1]

    p = intrinsic @ extrinsic @ p
    p /= p[2]
    p = p[:2].tolist()

    return p

# Inverse function of camera2img
def img2camera(image_point: Point2D) -> np.ndarray:
    """Implements perspective transformation. (page 58, Lecture 6)
    """
    # FIXME:
    # I negated the denominator `f - z` but not sure whether it is correct.
    # (lambda - f) >= 0 in lecture notes but (lambda - f) < 0 here.
    x, y = image_point
    z = arm_position[2] - 25

    point_img = np.array([x, y, z / -(f - z), 1])
    T = np.diag([1, 1, f, f / -(f - z)])
    point_camera = (T @ point_img)
    point_camera /= point_camera[-1]
    point_camera = point_camera[:-1]

    return point_camera[:2]

    # x_cam = ((x / f_x) * z) - (x_0 * z)
    # y_cam = ((y / f_y) * z) - (y_0 * z)

    # return (x_cam, y_cam)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary thresholding detector.")
    parser.add_argument('input', help='Input file name')
    args = parser.parse_args()

    img = cv2.imread(args.input)

    width, height = img.shape[1], img.shape[0]
    aspect_ratio = width / height
    downscale = 500 / height
    upscale = 1 / downscale

    objs, (gray, thresh, contours) = detect(img)
    for idx, obj in enumerate(objs, 1):
        cx, cy, phi = obj

        # Find the principle axis
        line = np.array(principleLine(width, obj))

        # Find coordinate related to robot base
        # Magic that idkw:
        #  - shift (cx, cy) before rotating and translating from image frame to camera frame. .
        #  - Either +85 or -85 (camera offset in camera frame)
        x, y, z = img2camera((cx - 640, cy - 480)).tolist()
        x, y, z, _ = T @ np.array([x, y + 85, z, 1])

        # Plot the centroid points and principle axis
        cv2.circle(img, (int(cx), int(cy)), 2, (255, 0, 255), thickness=4)
        plt.plot(line[:, 0], line[:, 1], (0, 0, 0), linewidth=0.25)
        plt.text(
            cx - 100, cy - 25, f"{idx}: {x:.0f}, {y:.0f}", #, {degrees(phi):.1f}°",
            c=(1, 1, 1), backgroundcolor=(0, 0, 0)
        )

    # z = 100
    # axis = [camera2img((400, y, z)) for y in range(250, 800, 25)] \
    #     + [camera2img((x, 250, z)) for x in range(400, 600, 25)]

    # axis = np.array(axis)

    plt.imshow(img)
    plt.show()
    plt.clf()

    plt.imshow(gray)
    plt.show()
    plt.clf()

    plt.imshow(thresh)
    plt.show()
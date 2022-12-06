import argparse
from math import atan2
from typing import List, Tuple

import cv2
import numpy as np

import detector
import path

# Type definition
Point2D = Tuple[float, float]  # (x, y)
Frame2D = Tuple[float, float, float]  # (x, y, phi)

# Homology transformation matrix
trans = np.array([
    [3.73555436e-01, -3.76758014e-01, 3.65298342e+02],
    [-3.74641915e-01, -3.76906441e-01, 8.38436436e+02],
    [-6.48402287e-07, -6.37826214e-06, 1.00000000e+00]
])


def img2world(image_point: Point2D) -> Point2D:
    """
    Use the homology transformation matrix to translate from pixel coordinates to world coordinates
    """
    x, y = image_point
    new_point = trans @ np.array([x, y, 1]).T
    x_w, y_w = new_point[:2]
    # Correct the scaling error in the transformation by adding the scaling error that is dependent on how far
    # the point is away from the image center (400, 400)
    x_error = (400 - x_w) * 0.04
    y_error = (400 - y_w) * 0.04
    return x_w + x_error, y_w + y_error


def find_contours(img):
    # Find contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect(img: np.ndarray) -> List[Frame2D]:
    """
    Detect objects based on binary thresholding method.
    Return the coordinate of central points (pixel) and principle angle (radian) in image frame.
    """

    contours = find_contours(img)

    # loop over the contours
    objects = []
    for c in contours:
        # Moments of the contour
        moment = cv2.moments(c)

        # Drop contour if the area is too small.
        if moment["m00"] < 600:
            continue

        # Object centroid
        c_x = moment["m10"] / moment["m00"]
        c_y = moment["m01"] / moment["m00"]

        # Principle angle 'phi'
        phi = atan2(2 * moment["mu11"], (moment["mu20"] - moment["mu02"])) / 2
        objects.append((c_x, c_y, phi))

    return objects


def calculate_path(cv2image):
    canvas, template = detector.get_sheets2(cv2image)

    path_points, jump_points = path.get_path(template)

    template_shape = template.shape

    template_dim1, template_dim2 = template_shape[:2]
    if template_dim1 >= template_dim2:
        template_corners = [(0, 0), (0, template_dim1), (template_dim2, template_dim1), (template_dim2, 0)]
    else:
        template_corners = [(template_dim2, 0), (0, 0), (0, template_dim1), (template_dim2, template_dim1)]

    c1, c2, c3, c4 = [img2world(corner) for corner in canvas]
    # c1, c2, c3, c4 = canvas
    dist1 = path.distance(c1, c2)
    dist2 = path.distance(c1, c4)
    distances = [dist1, dist2]
    distances.sort()

    if dist1 >= dist2:
        world_canvas_corners = [c1, c2, c3, c4]
    else:
        world_canvas_corners = [c2, c3, c4, c1]

    print(f"TEMPLATE: {template_corners}")
    print(f"CANVAS: {world_canvas_corners}")

    perspective_trans = cv2.getPerspectiveTransform(np.array(template_corners, np.float32),
                                                    np.array(world_canvas_corners, np.float32))

    pp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in path_points]
    jp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in jump_points]

    return pp, jp


def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    assert img is not None

    pp, jp = calculate_path(img)

    for p in pp:
        cv2.circle(img, (round(p[0]), round(p[1])), 5, (255, 0, 0), -1)

    cv2.imshow("path", img)

    print("DONE!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")

    return parser.parse_args()


if __name__ == "__main__":
    main()

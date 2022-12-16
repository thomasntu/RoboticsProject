import argparse
from math import atan2
from typing import List, Tuple

import cv2
import numpy as np

from . import detector, path

# import detector
# import path

# Type definition
Point2D = Tuple[float, float]  # (x, y)
Point3D = Tuple[float, float, float]  # (x, y)
Frame2D = Tuple[float, float, float]  # (x, y, phi)

# Homology transformation matrix
trans = np.array([
    [3.73555436e-01, -3.76758014e-01, 3.65298342e+02],
    [-3.74641915e-01, -3.76906441e-01, 8.38436436e+02],
    [-6.48402287e-07, -6.37826214e-06, 1.00000000e+00]
])

trans_3d = np.array([[1.14994841e+02, -1.07163016e+02, -1.98997675e+05, 1.02822512e+03],
                     [-1.07661695e+02, -9.99068873e+01, 1.86347863e+05, 1.11636878e+02],
                     [-2.54207649e-13, 9.33370067e-14, 3.46043044e-09, 1.00000000e+02]])

P1 = np.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])

P2 = np.array([[9.99992298e-01, -3.92471720e-03, 1.65555584e-05, 2.68678368e-02],
               [3.92471710e-03, 9.99992298e-01, 5.82079888e-06, 9.99638991e-01],
               [-1.65782758e-05, -5.75577816e-06, 1.00000000e+00, 8.06771556e-05]])


def stereo2world(image_point1: Point2D, image_point2: Point2D) -> Point3D:
    """
    Use the homology transformation matrix to translate from pixel coordinates to world coordinates
    """
    points_4d = cv2.triangulatePoints(P1, P2, [image_point1], [image_point2])

    x, y, z = points_4d[0]
    new_point = trans_3d @ np.array([x, y, z, 1]).T
    x_w, y_w, z_w = new_point[:3]

    return x_w, y_w, z_w


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


# Debugging function
def resize(img: np.ndarray, dim_limit=1920) -> np.ndarray:
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def calculate_path(cv2image, to_world=True, resize_dim=1920):
    """
    Returns
    -------
    pp: List[np.ndarray]
        path nodes

    jp: List[np.ndarray]
        jump nodes
    """
    resized = resize(cv2image, dim_limit=resize_dim)

    print("Finding canvas and template")
    canvas, template = detector.get_sheets2(resized)
    print(f"canvas: {canvas}")

    print("Calculating path")
    path_points, jump_points = path.get_path(template)

    template_dim1, template_dim2 = template.shape[:2]
    if template_dim1 >= template_dim2:
        # Template is vertical
        template_corners = [(0, 0), (0, template_dim1), (template_dim2, template_dim1), (template_dim2, 0)]
        print("case 1.1")
    else:
        # Template is horizontal
        template_corners = [(template_dim2, 0), (0, 0), (0, template_dim1), (template_dim2, template_dim1)]
        print("case 1.2")

    if to_world:
        c1, c2, c3, c4 = [img2world(corner) for corner in canvas]
    else:
        c1, c2, c3, c4 = canvas
    dist1 = path.distance(c1, c2)
    dist2 = path.distance(c1, c4)
    distances = [dist1, dist2]
    distances.sort()

    if dist1 >= dist2:
        # Canvas is horizontal
        world_canvas_corners = [c1, c2, c3, c4]
        print("case 2.1")
    else:
        # Canvas is vertical
        world_canvas_corners = [c2, c3, c4, c1]
        print("case 2.2")

    print(f"TEMPLATE: {template_corners}")
    print(f"CANVAS: {world_canvas_corners}")

    perspective_trans = cv2.getPerspectiveTransform(
        np.array(template_corners, np.float32),
        np.array(world_canvas_corners, np.float32)
    )

    pp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in path_points]
    jp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in jump_points]

    return pp, jp


def calculate_path_for_face(cv2image_face, cv2image_canvas, to_world=True, resize_dim=99999):
    """
    Returns
    -------
    pp: List[np.ndarray]
        path nodes

    jp: List[np.ndarray]
        jump nodes
    """
    resized_face = resize(cv2image_face, dim_limit=resize_dim)
    resized_canvas = resize(cv2image_canvas, dim_limit=resize_dim)

    print("Finding canvas and template")
    canvas, template = detector.face_detection(resized_face, resized_canvas)
    print(f"canvas: {canvas}")

    print("Calculating path")
    path_points, jump_points = path.get_path(template)

    template_dim1, template_dim2 = template.shape[:2]
    if template_dim1 >= template_dim2:
        # Template is vertical
        template_corners = [(0, 0), (0, template_dim1), (template_dim2, template_dim1), (template_dim2, 0)]
        print("case 1.1")
    else:
        # Template is horizontal
        template_corners = [(template_dim2, 0), (0, 0), (0, template_dim1), (template_dim2, template_dim1)]
        print("case 1.2")

    if to_world:
        c1, c2, c3, c4 = [img2world(corner) for corner in canvas]
    else:
        c1, c2, c3, c4 = canvas
    dist1 = path.distance(c1, c2)
    dist2 = path.distance(c1, c4)
    distances = [dist1, dist2]
    distances.sort()

    if dist1 >= dist2:
        # Canvas is horizontal
        world_canvas_corners = [c1, c2, c3, c4]
        print("case 2.1")
    else:
        # Canvas is vertical
        world_canvas_corners = [c2, c3, c4, c1]
        print("case 2.2")

    print(f"TEMPLATE: {template_corners}")
    print(f"CANVAS: {world_canvas_corners}")

    perspective_trans = cv2.getPerspectiveTransform(
        np.array(template_corners, np.float32),
        np.array(world_canvas_corners, np.float32)
    )

    pp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in path_points]
    jp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in jump_points]

    return pp, jp


def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    assert img is not None

    max_dim = 1080
    img = resize(img, max_dim)

    if args.output == "template":
        pp, jp = calculate_path(img, to_world=False, resize_dim=max_dim)

        for p in pp:
            cv2.circle(img, (round(p[0]), round(p[1])), 1, (255, 0, 0), -1)

        cv2.imshow("path", img)
    elif args.output == "face":
        # I/O and resize image if it's pretty large for GrabCut
        img_canvas = cv2.imread(args.input2)
        assert img_canvas is not None
        img_canvas = resize(img_canvas, max_dim)

        pp, jp = calculate_path_for_face(img, img_canvas, to_world=False, resize_dim=max_dim)

        for p in pp:
            cv2.circle(img_canvas, (round(p[0]), round(p[1])), 1, (255, 0, 0), -1)

        cv2.imshow("path", img_canvas)

    print("DONE!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")
    parser.add_argument('-i2', '--input2', help="Path to image 2")
    parser.add_argument('-o', '--output', help="template or face", default="template")

    return parser.parse_args()


if __name__ == "__main__":
    main()

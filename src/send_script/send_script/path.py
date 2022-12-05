import argparse
import math
from typing import List, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

Point2D = Tuple[float, float]  # (x, y)


def distance(a: Point2D, b: Point2D):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def find_nn(point: Point2D, points: Set[Point2D]):
    assert len(points) > 0
    min_dist = float('inf')
    nn = None
    for other in points:
        dist = distance(point, other)
        if dist < min_dist:
            min_dist = dist
            nn = other

    return nn, min_dist


def edge_detection_canny(cv_image):
    cv_image = cv2.flip(cv_image, 1)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # customize the second and the third argument, minVal and maxVal
    # in function cv2.Canny if needed
    get_edge = cv2.Canny(blurred, 10, 100)
    cv_image = np.hstack([get_edge])
    cv2.imshow('Corners_Extracted', cv_image)

    return cv_image


def path_pixels_to_points(cv_image) -> Set[Point2D]:
    [rows, cols] = np.shape(cv_image)
    print(f"The rows: {rows}")
    print(f"The cols: {cols}")
    # Create the arrays with the coordinate of the point that belongs to the corners detected
    points: Set[Point2D] = set()
    border = 10
    for i in range(border, rows - 1 - border):
        for j in range(border, cols - 1 - border):
            if cv_image[i, j] == 255:
                nx = j
                ny = i
                points.add((nx, ny))

    return points


def path_planning(_points: Set[Point2D]) -> Tuple[List[Point2D], List[Point2D]]:
    jumps: List[Point2D] = []
    path: List[Point2D] = []

    point = _points.pop()
    path.append(point)

    while len(_points) > 0:
        nn, dist = find_nn(point, _points)
        _points.remove(nn)
        path.append(nn)

        if dist > 3:
            jumps.append(point)

        point = nn

    return path, jumps


def straighten_lines(path, jumps):
    step = 2

    prev_points = path[1:step + 1]
    point0 = path[0]

    corners = [point0]

    old_angle = round(math.atan2(point0[1] - prev_points[-1][1], point0[0] - prev_points[-1][0]), 2)

    for point in path[step + 1:]:
        prev_point = prev_points[0]
        new_angle = round(math.atan2(prev_point[1] - point[1], prev_point[0] - point[0]), 2)

        jump = prev_point in jumps
        if jump or old_angle != new_angle:
            corners.append(prev_point)

        prev_points.append(point)
        prev_points = prev_points[1:]

        old_angle = new_angle

    corners.extend(prev_points[-step:])

    jump_diff = set(jumps) - set(corners)
    print(jump_diff)

    return corners


def detect_lines(cv2_image):
    lines = cv2.HoughLinesP(cv2_image, rho=1, theta=np.pi / 360, threshold=10, minLineLength=10, maxLineGap=1)
    color_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        line = line[0]
        point1 = (line[0], line[1])
        point2 = (line[2], line[3])
        cv2.line(img=color_image, pt1=point1, pt2=point2, color=(0, 255, 0), thickness=2)

    cv2.imshow("lines", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_path(cv2image) -> Tuple[List[Point2D], List[Point2D]]:
    corners_detected = edge_detection_canny(cv2image)
    points = path_pixels_to_points(corners_detected)
    path, jumps = path_planning(points)
    path = straighten_lines(path, jumps)
    return path, jumps


# Debugging function
def resize(img: np.ndarray) -> np.ndarray:
    dim_limit = 720

    max_dim = max(img.shape)
    resize_scale = dim_limit / max_dim
    img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    img = resize(img)

    points, jumps = get_path(img)

    prev_point = points[0]
    for corner in points[1:]:
        is_jump = prev_point in jumps
        plt.plot([prev_point[0], corner[0]], [prev_point[1], corner[1]], color='orange' if is_jump else 'blue')
        prev_point = corner

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y, color='yellow')

    plt.plot(x[0], y[0], color="green", marker="x")
    plt.plot(x[-1], y[-1], color="red", marker="x")

    print("DONE!")

    plt.show()
    print("Process finished correctly...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")

    return parser.parse_args()


if __name__ == "__main__":
    main()

import argparse
import math
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Definition of classes
class Point:  # Point definition
    def __init__(self, x, y, dist):
        self.x = x  # Coordinate x
        self.y = y  # Coordinate y
        self.dist = dist

    def __str__(self):  # Print the points
        return f'x: {self.x:.2f}, y: {self.y:.2f}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.dist == other.dist

    def __hash__(self):
        return hash(f"{self.x}{self.y}{self.dist}")

    def clone(self):
        return Point(self.x, self.y, self.dist)

    def calc_dist(self, val2):  # Distance from given point to other
        a = (val2.x - self.x)

        b = (val2.y - self.y)
        return math.sqrt(a * a + b * b)  # Pitagora's theorem

    def cn(self, _points):  # Looks for the closest neighbor
        aux = float('inf')  # Each time function is called, distance is supposed as infinity
        pos = 0  # Initialitation of index; also used for the last point
        for i in range(len(_points)):  # Checks the distance with all the points
            dist = self.calc_dist(_points[i])
            if (dist > 0) & (dist < aux):  # Keeps the point with minor distance
                aux = dist
                pos = i
                _points[i].dist = dist
        p = _points[pos]
        _points.pop(pos)
        return p


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


def path_planning_creation(cv_image) -> Tuple[List[Point], List[Point]]:
    [rows, cols] = np.shape(cv_image)
    print(f"The rows: {rows}")
    print(f"The cols: {cols}")
    # Create the arrays with the coordinate of the point that belongs to the corners detected
    points = []
    border = 10
    for i in range(border, rows - 1 - border):
        for j in range(border, cols - 1 - border):
            if cv_image[i, j] == 255:
                nx = j
                ny = i
                dist = float('inf')
                points.append(Point(nx, ny, dist))

    res = path_planning_with_greedy_colours(points)
    return res


def path_planning_with_greedy_colours(_points) -> Tuple[List[Point], List[Point]]:
    jumps = []
    path = []
    # Starts looking for the closest neighbour(CN)
    fp = _points[0]
    while True:
        fcn = fp.cn(_points)  # Calculate the CN

        if fcn.x == float('inf'):
            assert fcn.y == float('inf')
            break

        clone = fcn.clone()

        if fcn.dist > 4:
            jumps.append(fp.clone())
            jumps.append(clone)

        path.append(clone)

        fp.x = float('inf')  # Turns the points to infinity to not taking later
        fp.y = float('inf')
        fp = fcn

    return path, jumps


def straighten_lines(path, jumps):
    step = 2

    prev_points = path[1:step + 1]
    point0 = path[0]

    corners = [point0]

    old_angle = round(math.atan2(point0.y - prev_points[-1].y, point0.x - prev_points[-1].x), 2)

    for point in path[step + 1:]:
        prev_point = prev_points[0]
        new_angle = round(math.atan2(prev_point.y - point.y, prev_point.x - point.x), 2)

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


def get_path(cv2image) -> Tuple[List[Point], List[Point]]:
    """
    returns x and y coordinates of points + x and y coordinates of movements
    """
    print("Detecting Corners with Edge_Detection_Canny...")
    corners_detected = edge_detection_canny(cv2image)
    # detect_line(corners_detected)
    print("Creating Path_Planning ...")
    path, jumps = path_planning_creation(corners_detected)
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

    path_corners, jumps = get_path(img)

    x = [point.x for point in path_corners]
    y = [point.y for point in path_corners]

    x_jumps = [point.x for point in jumps]
    y_jumps = [point.y for point in jumps]

    plt.plot(x, y, color='blue')  # Plot the graph of points
    plt.scatter(x, y, color='yellow')  # Plot the graph of points

    plt.plot(x_jumps, y_jumps, color='orange')  # Plot the graph of point

    draw_start = (x[0], y[0])
    draw_end = (x[-1], y[-1])

    jump_start = (x_jumps[0], y_jumps[0])
    jump_end = (x_jumps[-1], y_jumps[-1])

    plt.plot(draw_start[0], draw_start[1], color="green", marker="x")
    plt.plot(draw_end[0], draw_end[1], color="red", marker="x")
    plt.plot(jump_start[0], jump_start[1], color="green", marker="+")
    plt.plot(jump_end[0], jump_end[1], color="red", marker="+")

    print(draw_start)
    print(draw_end)
    print(jump_start)
    print(jump_end)

    print("DONE!")

    plt.show()
    print("Process finished correctly...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")

    return parser.parse_args()


if __name__ == "__main__":
    main()

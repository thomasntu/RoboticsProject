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
    # cv2.waitKey(0)

    return cv_image


def path_planning_creation(cv_image) -> Tuple[List[Point], List[Point]]:
    [rows, cols] = np.shape(cv_image)
    print(f"The rows: {rows}")
    print(f"The cols: {cols}")
    # Create the arrays with the coordinate of the point that belongs to the corners detected
    points = []
    for i in range(rows - 1):
        for j in range(cols - 1):
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
    fcn = fp.cn(_points)
    for k in range(len(_points)):  # Looks for the CN of each point
        fp.x = float('inf')  # Turns the points to infinity to not taking later
        fp.y = float('inf')
        fp = fcn  # Take the CN as the starting point
        fcn = fp.cn(_points)  # Calculate the CN

        clone = fp.clone()

        if fp.dist > 4:
            jumps.append(clone)

        path.append(clone)

    return path, jumps


def straighten_lines(path, jumps):
    point0 = path[0]
    point1 = path[1]
    point2 = path[2]

    corners = [point0]

    old_angle = round(math.atan2(point0.y - point2.y, point0.x - point2.x), 2)

    for point3 in path[3:]:
        new_angle = round(math.atan2(point1.y - point3.y, point1.x - point3.x), 2)

        if point1 in jumps or old_angle != new_angle:
            corners.append(point1)

        point1 = point2
        point2 = point3
        old_angle = new_angle

    corners.append(point2)

    return corners


def get_path(cv2image) -> Tuple[List[Point], List[Point]]:
    """
    returns x and y coordinates of points + x and y coordinates of movements
    """
    print("Detecting Corners with Edge_Detection_Canny...")
    corners_detected = edge_detection_canny(cv2image)
    print("Creating Path_Planning ...")
    path, jumps = path_planning_creation(corners_detected)
    path = straighten_lines(path, jumps)
    return path, jumps


# Debugging function
def resize(img: np.ndarray) -> np.ndarray:
    dim_limit = 360

    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    img = resize(img)

    path_corners, jumps = get_path(img)

    for corner in path_corners[:10]:
        print(corner)

    x = [point.x for point in path_corners]
    y = [point.y for point in path_corners]

    x_jumps = [point.x for point in jumps]
    y_jumps = [point.y for point in jumps]

    plt.plot(x, y, color='blue')  # Plot the graph of points
    plt.scatter(x, y, color='yellow')  # Plot the graph of points
    plt.plot(x_jumps, y_jumps, color='red')  # Plot the graph of point

    print("DONE!")

    plt.show()
    print("Process finished correctly...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")

    return parser.parse_args()


if __name__ == "__main__":
    main()

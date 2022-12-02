import math
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def path_planning_creation(cv_image):
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
                nv = 1
                dist = float('inf')
                points.append(Point(nx, ny, nv, dist))

    res = path_planning_with_greedy_colours(points)
    return res


def path_planning_with_greedy_colours(_points):
    abcisas = []  # X drawn
    ordenadas = []  # Y drawn

    abcisasaux = []  # X move
    ordenadasaux = []  # Y move
    # Starts looking for the closest neighbour(CN)
    fp = _points[0]
    fcn = fp.cn(_points)
    for k in range(len(_points)):  # Looks for the CN of each point
        fp.x = float('inf')  # Turns the points to infinity to not taking later
        fp.y = float('inf')
        fp.v = 0
        fp = fcn  # Take the CN as the starting point
        fcn = fp.cn(_points)  # Calculate the CN
        if fp.dist > 4:
            abcisasaux.append(fp.x)
            ordenadasaux.append(fp.y)
        abcisas.append(fp.x)
        ordenadas.append(fp.y)

    return (abcisas, ordenadas), (abcisasaux, ordenadasaux)


# Definition of classes
class Point:  # Point definition
    def __init__(self, x, y, v, dist):
        self.x = x  # Coordinate x
        self.y = y  # Coordinate y
        self.v = v  # Vertex (1: yes ; 2: no)
        self.dist = dist

    def print_p(self):  # Print the points
        print('x: ' + str(self.x) + ', y:' + str(self.y) + ', v: ' + str(self.v) + ', dist: ' + str(self.dist))

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


def get_path(cv2image) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    """
    returns x and y coordinates of points + x and y coordinates of movements
    """
    print("Detecting Corners with Edge_Detection_Canny...")
    corners_detected = edge_detection_canny(cv2image)
    print("Creating Path_Planning ...")
    return path_planning_creation(corners_detected)


def main():
    img = cv2.imread("images/robot3.png")
    (x, y), (x_aux, y_aux) = get_path(img)
    plt.plot(x, y, color='blue')  # Plot the graph of points
    plt.plot(x_aux, y_aux, color='red')  # Plot the graph of point
    plt.show()
    print("Process finished correctly...")


if __name__ == "__main__":
    main()

import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Draw the points
abcisas = []  # X
ordenadas = []  # Y
distances = []
abcisasaux = []  # X
ordenadasaux = []  # Y
distancesaux = []


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

    path_planning_with_greedy_colours(points)
    cv2.imshow('Final Result', cv_image)
    # cv2.waitKey(0)


def path_planning_with_greedy_colours(_points):
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
            distancesaux.append(fp.dist)
        abcisas.append(fp.x)
        ordenadas.append(fp.y)
        distances.append(fp.dist)

    plt.plot(abcisas, ordenadas, color='blue')  # Plot the graph of points
    plt.plot(abcisasaux, ordenadasaux, color='red')  # Plot the graph of point
    plt.show()


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


def main():
    img = cv2.imread("images/robot3.png")
    print("Detecting Corners with Edge_Detection_Canny...")
    corners_detected = edge_detection_canny(img)
    print("Creating Path_Planning ...")
    path_planning_creation(corners_detected)
    [rows, cols, length] = np.shape(img)
    x = abcisas
    y = ordenadas
    dist = distances
    cols = cols
    rows = rows
    print("Process finished correctly...")
    print(f"X: {x}\n"
          f"Y: {y}\n"
          f"DIST: {dist}\n"
          f"COLS: {cols}\n"
          f"ROWS: {rows}\n")


if __name__ == "__main__":
    main()

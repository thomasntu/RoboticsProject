import argparse
import os
from math import atan2, degrees, tan
from typing import List, Tuple
import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect(img) -> List[Tuple[float, float, float]]:
    width, height = img.shape[1], img.shape[0]
    ratio = width / height
    dim = (int(500 * ratio), 500)

    img = cv2.resize(img, dim)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    objects = []
    for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        moment = cv2.moments(c)
        if moment["m00"] == 0:
            continue

        cX = int(moment["m10"] / moment["m00"])
        cY = int(moment["m01"] / moment["m00"])

        # Principle angle 'phi' and slope 'k' = 'tan(phi)'
        phi = atan2(2 * moment["mu11"], (moment["mu20"] - moment["mu02"])) / 2
        k = tan(phi)

        objects.append((cX, cY, phi))

        # y - cY = k(x - cX)
        # point1 = (0, -int(k * cX) + cY)
        # point2 = (int(500 * ratio), int(k * (500 * ratio - cX) + cY))
        # points = np.array([point1, point2])
        # print(points)

        # Plot the centroid points and principle axis
        # cv2.circle(img, (cX, cY), 2, (255, 0, 255), thickness=4)
        # plt.text(cX - 100, cY - 25, f"{cX}, {cY}, {degrees(phi):.1f}°", c=(1, 1, 1), backgroundcolor=(0, 0, 0))
        # plt.plot(points[:, 0], points[:, 1], (0, 0, 0))

    return objects

    # plt.imshow(img)

    # base = os.path.basename(args.input)
    # stem, ext = os.path.splitext(base)
    # plt.savefig(f"{stem}_classified{ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary image thresholding.")
    parser.add_argument('input', help='Input file name')
    args = parser.parse_args()

    img = cv2.imread(args.input)
    detect(img)

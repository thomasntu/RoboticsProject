# Source:
# https://learnopencv.com/automatic-document-scanner-using-opencv/

import argparse
import random
# import pysnooper
from typing import List, Optional

import cv2
import numpy as np


def order_points(pts) -> List[int]:
    """
    Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left

    Parameters
    ----------
    pts: array-like

    Returns
    -------
    rect: List[int]
    """
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()


# @pysnooper.snoop()
def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Finding the maximum height.
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Final destination co-ordinates.
    destination_corners = [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]]

    return order_points(destination_corners)


def find_rect(img: np.ndarray) -> List:
    """
    Find contours with simple content.
    """
    # Repeated closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

    # GrabCut: Remove background (Time consuming)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    roi = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, roi, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    # Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Finding contours for the detected edges.
    # Implementation: sort contours by area, and select 5 largest contours as candidates
    _, contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # OpenCV 3
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours[:2]  # only return outer contours


# @pysnooper.snoop()
def find_corner(contour) -> Optional[List[int]]:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) != 4:
        return None

    # Sorting the corners and converting array to desired shape.
    # np.concatenate() squeeze unused unused dimension: (4, 1, 2) -> (4, 2)
    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)

    return corners


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def find_templates_and_canvas(img, ):
    # TODO
    pass


# Debugging function
def resize(img: np.ndarray) -> np.ndarray:
    dim_limit = 1080

    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def get_rectangles(cv2image, draw_contours=False):
    # Find candidate contours and calculate corner if it can be approximated to rectangle
    contours = find_rect(cv2image)

    if draw_contours:
        for c in contours:
            cv2.drawContours(cv2image, c, -1,
                             (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)), 3)

    corners = [find_corner(c) for c in contours]
    rectangles = list(filter(lambda x: bool(x), corners))
    return rectangles


def get_sheets(cv2image) -> List[np.array]:
    rectangles = get_rectangles(cv2image)

    images = []
    for rect in rectangles:
        destination_corners = find_dest(rect)

        # Getting the homography and doing perspective transform.
        transformation = cv2.getPerspectiveTransform(np.float32(rect), np.float32(destination_corners))
        final = cv2.warpPerspective(
            cv2image, transformation, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        images.append(final)

    return images


# @pysnooper.snoop()
def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    img = resize(img)

    if args.output == "marked":

        rectangles = get_rectangles(img, draw_contours=True)

        # Displaying the contours and corners.
        for rect in rectangles:
            for char, corner in enumerate(rect, ord('A')):
                cv2.circle(img, tuple(corner), 5, (255, 0, 0), 2)
                cv2.putText(img, chr(char), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        print("DONE!")

        cv2.imshow("marked", img)

    elif args.output == "split":
        sheets = get_sheets(img)

        print("DONE!")

        for idx, sheet in enumerate(sheets):
            cv2.imshow(f"sheet{idx}", sheet)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")
    parser.add_argument('-o', '--output', help="split or marked", default="marked")

    return parser.parse_args()


if __name__ == "__main__":
    main()

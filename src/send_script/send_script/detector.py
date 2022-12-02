# Source:
# https://learnopencv.com/automatic-document-scanner-using-opencv/

import argparse
import cv2
import numpy as np
import pysnooper
from typing import List, Tuple, Optional


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


@pysnooper.snoop()
def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def find_rect(img: np.ndarray) -> Tuple:
    """
    Find contours with simple content.
    """
    # Repeated closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

    # GrabCut: Remove background (Time consuming)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    roi = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, roi, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
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
    # contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


@pysnooper.snoop()
def find_corner(contour) -> List[int]:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) != 4:
        return None

    # Sorting the corners and converting array to desired shape.
    # np.concatenate() squeeze unused unused dimension: (4, 1, 2) -> (4, 2)
    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)

    return corners


def resize(img: np.ndarray) -> np.ndarray:
    dim_limit = 1080

    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")

    return parser.parse_args()


def get_corners(cv2image):
    # Find candidate contours and calculate corner if it can be approximated to rectangle
    contours = find_rect(cv2image)

    for c in contours:
        cv2.drawContours(cv2image, c, -1, (0, 255, 255), 3)

    corners = [find_corner(c) for c in contours]
    rectangles = filter(lambda x: bool(x), corners)
    return rectangles


def get_sheets(cv2image) -> List[np.array]:
    rectangles = get_corners(cv2image)

    images = []
    for rect in rectangles:
        destination_corners = find_dest(rect)

        # Getting the homography and doing perspective transform.
        T = cv2.getPerspectiveTransform(np.float32(rect), np.float32(destination_corners))
        final = cv2.warpPerspective(
            cv2image, T, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        images.append(final)

    return images


@pysnooper.snoop()
def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    img = resize(img)

    rectangles = get_corners(img)

    # Displaying the contours and corners.
    for rect in rectangles:
        for char, corner in enumerate(rect, ord('A')):
            cv2.circle(img, tuple(corner), 5, (255, 0, 0), 2)
            cv2.putText(img, chr(char), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite('Image.png', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()

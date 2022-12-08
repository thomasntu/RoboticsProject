# Source:
# https://learnopencv.com/automatic-document-scanner-using-opencv/

import argparse
# import pysnooper
from typing import List, Optional, Tuple

import cv2
import numpy as np


def order_points(pts) -> List[Tuple[int, int]]:
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
    """
    Used by cv2.warpPerspective(), to perserve the aspect ratio of target.
    """
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


def find_contours(img: np.ndarray) -> List:
    """Find contours with simple content.
    """
    # Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # cv2.imshow("blurred", blurred)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    # v2.imshow("thresh", thresh)
    # canny = cv2.Canny(thresh, 0, 200)
    # cv2.imshow("canny", canny)
    # dilated = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # cv2.imshow("dilated", dilated)

    # Finding contours for the detected edges.
    # Implementation:
    # - RETR_EXTERNAL: Keep the external contours and discard all internal contours
    # - sort contours with area in descending order
    # _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 3
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


# @pysnooper.snoop()
def find_contour_with_n_corner(contour, n=4) -> Optional[List[Tuple[int, int]]]:
    """Approximate contour with `n` corners.
    """
    # Don't forget another method cv2.convexHull() and cv2.minAreaRect()
    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) != n:
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


def get_sheets2(cv2image: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Returns
    -------
    coordinate: Tuple[int, int]

    pattern: np.ndarray
    """
    # Find candidate contours and calculate corner if it can be approximated to rectangle
    contours = find_contours(cv2image)

    rectangles = []
    for contour in contours:
        corner = find_contour_with_n_corner(contour)
        if corner is not None:
            rectangles.append(corner)

    rectangles = rectangles[:2]  # only use largest 2 rectangles

    # Classify if it is pattern or canvas
    images = []
    for idx, rect in enumerate(rectangles):
        destination_corners = find_dest(rect)
        w, h = destination_corners[2]

        # Getting the homography and doing perspective transform.
        transformation = cv2.getPerspectiveTransform(np.float32(rect), np.float32(destination_corners))
        sheet = cv2.warpPerspective(cv2image, transformation, (w, h), flags=cv2.INTER_LINEAR)

        # Find canvas if it has no edges on it
        edge = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(edge, 0, 200)
        count = np.bincount(edge.flatten())
        ratio = count[0] / (w * h)

        images.append((ratio, rect, sheet))

    assert len(images) == 2

    def get_ratio(x):
        return x[0]

    images = sorted(images, key=get_ratio, reverse=True)
    return images[0][1], images[1][2]


# Debugging function
def resize(img: np.ndarray) -> np.ndarray:
    dim_limit = 1080

    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def get_rectangles(cv2image, draw_contours=False) -> List[List[int]]:
    # Find candidate contours and calculate corner if it can be approximated to rectangle
    contours = find_contours(cv2image)

    if draw_contours:
        for c in contours:
            cv2.drawContours(cv2image, c, -1, (0, 255, 0), 3)

    corners = [find_contour_with_n_corner(c) for c in contours]
    rectangles = list(filter(lambda x: bool(x), corners))
    return rectangles[:2]


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


def get_single_canvas(cv2image: np.ndarray) -> List[Tuple[int, int]]:
    # Find candidate contours and calculate corner if it can be approximated to rectangle
    contours = find_contours(cv2image)

    rectangles = []
    for contour in contours:
        corner = find_contour_with_n_corner(contour)
        if corner is not None:
            rectangles.append(corner)

    return rectangles[0]  # only use the largest rectangle


def face_detection(cv2image_face: np.ndarray, cv2image_canvas: np.ndarray):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(cv2image_face, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    (x, y, w, h) = faces[0]
    face = cv2image_face.copy()[y:y + h, x:x + w]
    canvas = get_single_canvas(cv2image_canvas)
    return canvas, face


# @pysnooper.snoop()
def main():
    args = parse_args()

    # I/O and resize image if it's pretty large for GrabCut
    img = cv2.imread(args.input)
    assert img is not None
    # img = resize(img)

    if args.output == "marked":

        rectangles = get_rectangles(img, draw_contours=True)

        # Displaying the contours and corners.
        for rect in rectangles:
            for char, corner in enumerate(rect, ord('A')):
                cv2.circle(img, tuple(corner), 5, (255, 0, 0), 2)
                cv2.putText(img, chr(char), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        print("DONE!")

        cv2.imshow("marked", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.output == "split":
        sheets = get_sheets(img)

        for idx, sheet in enumerate(sheets):
            cv2.imwrite(f"images/sheet{idx}.png", sheet)

    elif args.output == "prod":
        canvas, template = get_sheets2(img)

        for c in canvas:
            cv2.circle(img, c, 5, (255, 0, 0), -1)

        cv2.imshow("scene", img)
        cv2.imshow("pattern", template)

        print("DONE!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.output == "face":
        img_canvas = cv2.imread(args.input2)
        assert img_canvas is not None

        img = resize(img)
        img_canvas = resize(img_canvas)

        canvas, template = face_detection(img, img_canvas)

        for c in canvas:
            cv2.circle(img_canvas, c, 10, (255, 0, 0), -1)
            cv2.line(img_canvas, (0, 0), c, (0, 255, 0), 1)

        cv2.imshow("canvas", img_canvas)
        cv2.imshow("face", template)

        print("DONE!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")
    parser.add_argument('-i2', '--input2', help="Path to image 2")
    parser.add_argument('-o', '--output', help="split, marked, face or prod", default="prod")

    return parser.parse_args()


if __name__ == "__main__":
    main()

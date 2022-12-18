import argparse
import math
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]  # (x, y)
Corner = Tuple[int, int]  # (x, y)
Rectangle = List[Corner]  # [c1, c2, c3, c4]
Image = np.ndarray


def img2world(image_point: Point2D) -> Point2D:
    """
    Use the homology transformation matrix to translate from pixel coordinates to world coordinates
    """
    # Homology transformation matrix
    trans = np.array([
        [3.73555436e-01, -3.76758014e-01, 3.65298342e+02],
        [-3.74641915e-01, -3.76906441e-01, 8.38436436e+02],
        [-6.48402287e-07, -6.37826214e-06, 1.00000000e+00]
    ])

    x, y = image_point
    new_point = trans @ np.array([x, y, 1]).T
    x_w, y_w = new_point[:2]
    # Correct the scaling error in the transformation by adding the scaling error that is dependent on how far
    # the point is away from the image center (400, 400)
    x_error = (400 - x_w) * 0.04
    y_error = (400 - y_w) * 0.04
    return x_w + x_error, y_w + y_error


def find_contours(img: Image) -> List[List[Point2D]]:
    """
    Find contours with simple content.
    """
    # Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 3
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def order_points(pts: Rectangle) -> Rectangle:
    """
    Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left
    """
    rect: Rectangle = [0, 0, 0, 0]
    pts = np.array(pts)

    sum = np.sum(pts, axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(sum)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(sum)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect


def approx_rectangle(contour) -> Optional[Rectangle]:
    """
    Approximate contour with 4 corners.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) != 4:
        return None

    # Sorting the corners and converting array to desired shape.
    # np.concatenate() squeeze unused unused dimension: (4, 1, 2) -> (4, 2)
    corners = np.concatenate(corners).tolist()
    corners = order_points(corners)

    return corners


def get_sheets(cv2image: np.ndarray, n: int) -> List[Rectangle]:
    """
    Return n largest rectangles in image
    """
    # Find candidate contours and calculate corner if it can be approximated to rectangle
    contours = find_contours(cv2image)

    rectangles = []
    for contour in contours:
        corner = approx_rectangle(contour)
        if corner is not None:
            rectangles.append(corner)

    rectangles = rectangles[:n]  # only use largest n rectangles
    return rectangles


def find_dest(rectangle: Rectangle) -> Rectangle:
    """
    Find the destination rectangle to warp the original rectangle to.
    The destination rectangle has a maximum width and height, and the corners are ordered top-left, top-right,
    bottom-right, bottom-left.
    """
    (tl, tr, br, bl) = rectangle
    # Finding the maximum width.
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Finding the maximum height.
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Final destination coordinates.
    destination_corners = [(0, 0), (max_width, 0), (max_width, max_height), (0, max_height)]

    return order_points(destination_corners)


def find_canvas_and_template(cv2image: Image, rectangles: List[Rectangle]) -> Tuple[Rectangle, Image]:
    """
    Given 2 rectangles, find the canvas and template rectangles in the image.
    The canvas rectangle is the one with no edges, while the template rectangle is the one with edges.
    Returns a tuple containing the canvas rectangle and the template image.
    """
    assert len(rectangles) == 2

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

    def get_ratio(x):
        return x[0]

    images = sorted(images, key=get_ratio, reverse=True)
    return images[0][1], images[1][2]


def get_face(cv2image: Image):
    """
    Detect and return the face in the given image.
    """
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    (x, y, w, h) = faces[0]
    face = cv2image.copy()[y:y + h, x:x + w]
    return face


def edge_detection_canny(cv2image: Image):
    """
    Perform edge detection on the given image using the Canny algorithm.
    Returns the image with edges highlighted.
    """
    img = cv2.flip(cv2image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1, 1)
    img = cv2.Canny(img, 10, 100)

    return img


def path_pixels_to_points(cv_image) -> Set[Point2D]:
    """
    Convert the white pixels in the given image to points in a set.
    """
    [rows, cols] = np.shape(cv_image)
    margin = round(min(rows, cols) * 0.1)
    # Create the arrays with the coordinate of the point that belongs to the corners detected
    points: Set[Point2D] = set()
    for i in range(margin, rows - 1 - margin):
        for j in range(margin, cols - 1 - margin):
            if cv_image[i, j] == 255:
                nx = j
                ny = i
                points.add((nx, ny))

    return points


def distance(a: Point2D, b: Point2D):
    """
    Calculates the Euclidean distance between two points in 2D space.
    """
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def find_nn(point: Point2D, points: Set[Point2D]):
    """
    Finds the nearest neighbor (NN) of a given point among a set of points using the Euclidean distance metric.
    Returns the NN point and the distance between the two points.
    """
    assert len(points) > 0
    min_dist = float('inf')
    nn = None
    for other in points:
        dist = distance(point, other)
        if dist < min_dist:
            min_dist = dist
            nn = other

    return nn, min_dist


def path_planning(points: Set[Point2D]) -> Tuple[List[Point2D], List[Point2D]]:
    """
    Plans a path through a set of 2D points, identifying "jumps" (points more than 2 units away from their
    nearest neighbor) along the way.
    Returns the path as a list of points and a list of jump points.
    """
    jumps: List[Point2D] = []
    path: List[Point2D] = []

    point = points.pop()

    path_segment: List[Point2D] = [point]
    while len(points) > 0:
        nn, dist = find_nn(point, points)
        points.remove(nn)

        if dist > 2:
            if len(path_segment) > 5:
                jumps.append(path_segment[-1])
                path.extend(path_segment)
            path_segment.clear()

        path_segment.append(nn)
        point = nn

    return path, jumps


def straighten_lines(path, jumps) -> List[Point2D]:
    """
    Modifies a path by identifying and removing unnecessary points on lines (points where the direction of the path
    does not change).
    Jumps (points more than 2 units away from their nearest neighbor) are preserved as points.
    Returns the modified path as a list of points.
    """
    step = 2
    prev_points = path[1:step + 1]
    point0 = path[0]

    corners = [point0]

    old_angle = round(math.atan2(point0[1] - prev_points[-1][1], point0[0] - prev_points[-1][0]), 2)

    for point in path[step + 1:]:
        prev_point = prev_points[0]
        new_angle = round(math.atan2(prev_point[1] - point[1], prev_point[0] - point[0]), 1)

        jump = prev_point in jumps
        if jump or old_angle != new_angle:
            corners.append(prev_point)

        prev_points.append(point)
        prev_points = prev_points[1:]

        old_angle = new_angle

    corners.extend(prev_points[-step:])

    return corners


def get_path(cv2image: Image) -> Tuple[List[Point2D], List[Point2D]]:
    """
    Generates a path through an image.
    Returns the path as a list of points and a list of
    "jump" points (points more than 2 units away from their nearest neighbor).
    """
    corners_detected = edge_detection_canny(cv2image)
    points = path_pixels_to_points(corners_detected)
    path, jumps = path_planning(points)
    path = straighten_lines(path, jumps)
    return path, jumps


def copy_image(cv2image: Image, to_world=True) -> Tuple[List[Point2D], List[Point2D]]:
    """
    Generates a path for copying an image from a canvas onto a template.
    Returns the path as a list of points.
    If to_world is True, the path is scaled to world coordinates. Otherwise, it is left in pixel coordinates.
    """
    sheets = get_sheets(cv2image, 2)
    canvas, template = find_canvas_and_template(cv2image, sheets)
    return calculate_path(canvas, template, to_world=to_world)


def draw_face(cv2image_face: Image, cv2image_canvas: Image, to_world=True) -> Tuple[List[Point2D], List[Point2D]]:
    """
    Generates a path for drawing a face from a source image onto a canvas.
    Returns the path as a list of points.
    If to_world is True, the path is scaled to world coordinates. Otherwise, it is left in pixel coordinates.
    """
    [canvas] = get_sheets(cv2image_canvas, 1)
    template = get_face(cv2image_face)
    return calculate_path(canvas, template, to_world=to_world)


def calculate_path(canvas, template, to_world=True) -> Tuple[List[Point2D], List[Point2D]]:
    """
    Transforms a path from a template image to world coordinates or pixel coordinates on a canvas image.
    Returns the transformed path as a list of points.
    If to_world is True, the path is scaled to world coordinates. Otherwise, it is left in pixel coordinates.
    """
    resized_template = resize(template, 720)
    path_points, jump_points = get_path(resized_template)

    template_dim1, template_dim2 = resized_template.shape[:2]
    if template_dim1 >= template_dim2:
        # Template is vertical
        template_corners = [(0, 0), (0, template_dim1), (template_dim2, template_dim1), (template_dim2, 0)]
    else:
        # Template is horizontal
        template_corners = [(template_dim2, 0), (0, 0), (0, template_dim1), (template_dim2, template_dim1)]

    if to_world:
        c1, c2, c3, c4 = [img2world(corner) for corner in canvas]
    else:
        c1, c2, c3, c4 = canvas
    dist1 = distance(c1, c2)
    dist2 = distance(c1, c4)
    distances = [dist1, dist2]
    distances.sort()

    if dist1 >= dist2:
        # Canvas is horizontal
        world_canvas_corners = [c1, c2, c3, c4]
    else:
        # Canvas is vertical
        world_canvas_corners = [c2, c3, c4, c1]

    perspective_trans = cv2.getPerspectiveTransform(
        np.array(template_corners, np.float32),
        np.array(world_canvas_corners, np.float32)
    )

    pp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in path_points]
    jp = [(perspective_trans @ np.array([point[0], point[1], 1]))[:2] for point in jump_points]

    return pp, jp


def resize(img: Image, dim_limit=720) -> Image:
    """
    Resizes an image so that the maximum dimension (either width or height) does not exceed a given limit.
    Returns the resized image.
    """
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    return img


def main():
    """
    Main function that reads an input image, processes it, and displays the result.
    If the output command line argument is "copy", the function generates a path for copying the input image.
    If the output command line argument is "face", the function generates a path for drawing a face from a second
    input image onto the canvas.
    The image is resized if its maximum dimension exceeds 720 pixels.
    """
    args = parse_args()

    img: Image = cv2.imread(args.input)
    assert img is not None
    img = resize(img)

    if args.output == "copy":
        pp, jp = copy_image(img, to_world=False)

    elif args.output == "face":
        face: Image = cv2.imread(args.input2)
        face = resize(face)
        assert face is not None
        pp, jp = draw_face(face, img, to_world=False)

    for p in pp:
        cv2.circle(img, (round(p[0]), round(p[1])), 1, (255, 0, 0), -1)

    cv2.imshow("path", img)

    print("DONE!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments using argparse.
    Returns an object with the parsed arguments as attributes.
    """
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")
    parser.add_argument('-i2', '--input2', help="Path to face image")
    parser.add_argument('-o', '--output', help="copy or face", default="copy")

    return parser.parse_args()


if __name__ == "__main__":
    main()

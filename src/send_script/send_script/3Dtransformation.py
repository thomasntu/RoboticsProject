import cv2
import numpy as np


def get_trans(pixel_coords1, pixel_coords2, world_coords):
    # Set up the pixel and world coordinate systems
    # pixel_coords1 = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.float32)
    # pixel_coords2 = np.array([[15, 25], [35, 45], [55, 65], [75, 85]], dtype=np.float32)
    # world_coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

    # Estimate the essential matrix using cv2.findEssentialMat()
    E, _ = cv2.findEssentialMat(pixel_coords1, pixel_coords2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999,
                                threshold=1.0)

    # Recover the relative rotation and translation using cv2.recoverPose()
    _, R, t, _ = cv2.recoverPose(E, pixel_coords1, pixel_coords2, focal=1.0, pp=(0., 0.))

    # Set up the projection matrices for both cameras
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # projection matrix for camera 1
    P2 = np.hstack((R, t))  # projection matrix for camera 2

    print(f"P1: {P1}")
    print(f"P2: {P2}")

    # Compute the 3D world coordinates of the points using cv2.triangulatePoints()
    points_4d = cv2.triangulatePoints(P1, P2, pixel_coords1, pixel_coords2)
    # print(f"points_4d: {points_4d}")
    points_4d /= points_4d[3]  # normalize homogeneous coordinates
    world_coords_converted = points_4d[:3].T

    print(world_coords)  # should print the 3D world coordinates corresponding to the pixel coordinates in
    print(world_coords_converted)  # should print the 3D world coordinates corresponding to the pixel coordinates in

    _, trans, _ = cv2.estimateAffine3D(world_coords_converted, world_coords)

    return trans


def find_image_points(_file_name, draw_image_points=False):
    """
    Fin the image points on the picture of a checkerboard.
    :param _file_name: filename of the image
    :param draw_image_points: save the image with the corners drawn onto it
    :return: tuple with boolean if process was successful and image points if it was
    """
    _imgp = None
    image = cv2.imread(_file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    if success:
        _imgp = cv2.cornerSubPix(gray,
                                 corners,
                                 (11, 11),
                                 (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        if draw_image_points:
            cv2.drawChessboardCorners(image, (8, 6), _imgp, True)
            cv2.imwrite(f'{_file_name[:-4]}-corners.png', image)

    else:
        print('FAILED')

    return success, _imgp


def main():
    # Find corners for all distorted images
    file_name1 = "checkers/3D/IMG_1_noflash.png"
    file_name2 = "checkers/3D/IMG_2_noflash.png"
    # Find corners
    _, imgp1 = find_image_points(file_name1, draw_image_points=True)
    _, imgp2 = find_image_points(file_name2, draw_image_points=True)

    # print(imgp1)
    # print(imgp2)

    y_p = 485.4
    x_p = 461
    # Construct the real world coordinates of all other corners
    objp = np.zeros((6 * 8, 3), np.float32)
    idx = 0
    for x in range(0, 6):
        for y in range(0, 8):
            objp[idx][0] = x_p - x * 24.4
            objp[idx][1] = y_p - y * 24.4
            objp[idx][2] = 100
            idx += 1

    trans = get_trans(imgp1, imgp2, objp)

    print(trans)


if __name__ == "__main__":
    main()

# center of checkerboard 400, 400
# checker width 24.4
# imag positions 1: 300, 300; 2: 400, 400

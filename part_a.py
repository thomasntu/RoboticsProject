import argparse
import glob

import cv2 as cv
import numpy as np


def main(_args: argparse.Namespace):
    set_idx = _args.set
    # Checkerboard grid
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Array of checkerboard grids
    object_points = []
    # Array of checkerboard corners found in distorted images
    image_points = []
    # Find all PNG files in folder
    file_names = glob.glob(f'checker*.png')
    print(file_names)
    # Save an image with found corners for the first image
    draw_corners = True
    # Find corners for all distorted images
    for file_name in file_names:
        # Find corners
        ret, imgp = find_image_points(file_name, draw_image_points=draw_corners)
        if ret:
            # If corners are found, append to arrays
            object_points.append(objp)
            image_points.append(imgp)
            # Draw corners only for first successful image
            draw_corners = False

    # Get camera parameters
    shape = cv.imread(file_names[0]).shape
    _, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv.calibrateCamera(object_points,
                                                                                                          image_points,
                                                                                                          shape[
                                                                                                          -2:-4:-1],
                                                                                                          None,
                                                                                                          None)

    # # Do not crop first image
    # save_uncropped = True
    # # Undistort checkerboard images
    # for file_name in file_names:
    #     undistort(file_name, camera_matrix, distortion_coefficients, _save_uncropped=save_uncropped)
    #     save_uncropped = False

    # # Undistort extra image
    # extra_img = glob.glob(f'extra{set_idx}.png')[0]
    # undistort(extra_img, camera_matrix, distortion_coefficients, _save_uncropped=True)

    # Print camera matrix and
    print()
    print(f"Camera Matrix:\n{camera_matrix}")
    print()
    print(f"Distortion Coefficients:\n{distortion_coefficients}")
    print()
    h, w = shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    print(f"New Camera Matrix:\n{new_camera_mtx}")
    print()
    print(f"ROI:\n{roi}")
    print()


def undistort(_file_name, _mtx, _dist, _save_uncropped=False):
    """
    Undistort the image with the given filename using the given camera parameters.
    :param _file_name: filename of the image
    :param _mtx: camera matrix
    :param _dist: distortion coefficients
    :param _save_uncropped: save uncropped image
    :return: None
    """
    _img = cv.imread(_file_name)
    h, w = _img.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(_mtx, _dist, (w, h), 1, (w, h))
    dst = cv.undistort(_img, _mtx, _dist, None, new_camera_mtx)

    # if _save_uncropped:
    #     cv.imwrite(f'undistorted/{_file_name[:-4]}-uncropped.png', dst)

    # Crop image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv.imwrite(f'undistorted/{_file_name[:-4]}-undistorted.png', dst)


def find_image_points(_file_name, draw_image_points=False):
    """
    Fin the image points on the picture of a checkerboard.
    :param _file_name: filename of the image
    :param draw_image_points: save the image with the corners drawn onto it
    :return: tuple with boolean if process was successful and image points if it was
    """
    _imgp = None
    image = cv.imread(_file_name)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    success, corners = cv.findChessboardCorners(gray, (8, 6), None)
    if success:
        _imgp = cv.cornerSubPix(gray,
                                corners,
                                (11, 11),
                                (-1, -1),
                                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    else:
        print('FAILED')
        # if draw_image_points:
        #     cv.drawChessboardCorners(image, (8, 6), _imgp, True)
        #     cv.imwrite(f'{_file_name[:-4]}-corners.png', image)
    return success, _imgp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration.")
    parser.add_argument("-s", "--set", help='Input images set (1 or 2)', default=2, required=False, type=int)
    args = parser.parse_args()

    main(args)

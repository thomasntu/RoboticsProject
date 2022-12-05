import cv2 as cv
import numpy as np

# We lost the original picture of the checkerboard, but these are the coordinates of the detected corners
_corners = np.array([
    [[599.0467, 472.47952]],
    [[632.2732, 505.668]],
    [[665.42224, 538.77094]],
    [[698.5991, 572.0023]],
    [[731.87585, 605.2216]],
    [[765.1736, 638.5072]],
    [[798.50775, 671.7587]],
    [[831.74506, 705.09357]],
    [[565.72845, 505.63727]],
    [[599.12085, 538.84546]],
    [[632.26294, 571.93616]],
    [[665.47174, 605.1649]],
    [[698.6637, 638.47095]],
    [[731.9851, 671.76514]],
    [[765.24097, 705.1047]],
    [[798.603, 738.3983]],
    [[532.51526, 538.9009]],
    [[565.7968, 572.1681]],
    [[599.0493, 605.22876]],
    [[632.2838, 638.4315]],
    [[665.4928, 671.74457]],
    [[698.7607, 704.92285]],
    [[731.9378, 738.2512]],
    [[765.1651, 771.43195]],
    [[499.1896, 572.2818]],
    [[532.5526, 605.53876]],
    [[565.7682, 638.535]],
    [[599.00385, 671.66296]],
    [[632.2927, 705.0535]],
    [[665.4845, 738.1284]],
    [[698.65704, 771.235]],
    [[731.8771, 804.5187]],
    [[465.8316, 605.582]],
    [[499.1712, 638.85944]],
    [[532.53546, 671.7965]],
    [[565.6827, 705.05054]],
    [[598.9731, 738.41003]],
    [[632.25946, 771.268]],
    [[665.4476, 804.3312]],
    [[698.5923, 837.584]],
    [[432.35083, 638.91327]],
    [[465.88123, 672.16046]],
    [[499.18835, 705.37573]],
    [[532.52545, 738.37366]],
    [[565.67426, 771.58514]],
    [[599.02277, 804.594]],
    [[632.20764, 837.5829]],
    [[665.4334, 870.6653]]
], np.float32)


def main():
    # Real world coordinates of the first corner
    y_p = 437.5
    x_p = 412.5

    # Construct the real world coordinates of all other corners
    objp = np.zeros((6 * 8, 1, 2), np.float32)
    idx = 0
    for x in range(0, 6):
        for y in range(0, 8):
            objp[idx][0][0] = x_p - x * 25
            objp[idx][0][1] = y_p - y * 25
            idx += 1

    # Find corners for all distorted images
    file_name = "images/IMG.png"
    # Find corners
    # _, imgp = find_image_points(file_name)
    imgp = _corners

    print(imgp)
    print(objp)

    ret, _ = cv.findHomography(imgp, objp)

    print(ret)


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

        if draw_image_points:
            cv.drawChessboardCorners(image, (8, 6), _imgp, True)
            cv.imwrite(f'{_file_name[:-4]}-corners.png', image)

    else:
        print('FAILED')

    return success, _imgp


if __name__ == "__main__":
    main()

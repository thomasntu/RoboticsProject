import glob
from math import degrees
from re import T

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R

rows = 6 #number of checkerboard rows.
columns = 8 #number of checkerboard columns.
world_scaling = 2.4 #change this to the real world square size. Or not.

def find_chessboard(imgs):
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    # Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for i, frame in enumerate(imgs):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f'{i}', gray)

        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

    return np.squeeze(np.array(objpoints)), np.squeeze(np.array(imgpoints))


def triangulate(p1, p2, mtx1, mtx2, R, T):
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1

    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2

    def DLT(P1, P2, point1, point2):
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        B = A.transpose() @ A

        _, _, Vh = linalg.svd(B, full_matrices = False)

        return Vh[3,0:3] / Vh[3,3]

    p3ds = []
    for uv1, uv2 in zip(p1, p2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)

    return np.array(p3ds)

def main():
    imgs = [
        cv2.imread('../../../images/stereo2/IMG013153.png'),
        cv2.imread('../../../images/stereo2/IMG013155.png'),
        cv2.imread('../../../images/stereo2/IMG013156.png'),
        cv2.imread('../../../images/stereo2/IMG013158.png'),
        cv2.imread('../../../images/stereo2/IMG013159.png'),
        cv2.imread('../../../images/stereo2/IMG013201.png'),
    ]
    height, width = imgs[0].shape[0:2]

    frames = [
        (R.from_euler('xyz', [-180, 0, 135], degrees=True).as_rotvec(), (np.array([275, 425, 730])).T),
        (R.from_euler('xyz', [-185, 0, 135], degrees=True).as_rotvec(), (np.array([350, 350, 730])).T),
        (R.from_euler('xyz', [-190, 0, 135], degrees=True).as_rotvec(), (np.array([350, 350, 730])).T),
        (R.from_euler('xyz', [-175, 0, 135], degrees=True).as_rotvec(), (np.array([350, 350, 730])).T),
        (R.from_euler('xyz', [-170, 0, 135], degrees=True).as_rotvec(), (np.array([350, 350, 730])).T),
        (R.from_euler('xyz', [-180, 0, 135], degrees=True).as_rotvec(), (np.array([425, 275, 730])).T),
    ]
    rots, tvecs = [f[0] for f in frames], [f[1] for f in frames]

    objp, imgp = find_chessboard(imgs)
    ret, CM1, dist1, r, t = cv2.calibrateCamera(objp, imgp, (width, height), None, None)

    r, t = cv2.calibrateHandEye(
        rots, tvecs, r, t, cv2.CALIB_HAND_EYE_TSAI
    )
    print(CM1, dist1)
    print(r, t)

    # _, imgp2 = find_chessboard(images2)
    # ret, mtx2, dist2, _, _ = cv2.calibrateCamera([objp], [imgp2], (width, height), None, None)

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    # ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
    #     [objp], [imgp1], [imgp2], mtx1, dist1, mtx2, dist2, (width, height),
    #     criteria = criteria, flags = cv2.CALIB_FIX_INTRINSIC
    # )


if __name__ == "__main__":
    main()

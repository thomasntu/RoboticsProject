import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy import linalg

rows = 6 #number of checkerboard rows.
columns = 8 #number of checkerboard columns.
world_scaling = 2.4 #change this to the real world square size. Or not.

orientation1 = 0
orientation2 = 0

def find_chessboard(images):
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

    for i, frame in enumerate(images):
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
    images = [cv2.imread(imname, 1) for imname in glob.glob('../../../images/stereo/orientation1-350-350-730/*.png')]
    height, width = images[0].shape[0:2]

    objp, imgp1 = find_chessboard(images)
    ret, mtx1, dist1, _, _ = cv2.calibrateCamera(objp, imgp1, (width, height), None, None)

    images = [cv2.imread(imname, 1) for imname in glob.glob('../../../images/stereo/orientation2-300-300-730/*.png')]

    _, imgp2 = find_chessboard(images)
    ret, mtx2, dist2, _, _ = cv2.calibrateCamera(objp, imgp2, (width, height), None, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objp, imgp1, imgp2, mtx1, dist1, mtx2, dist2, (width, height),
        criteria = criteria, flags = cv2.CALIB_FIX_INTRINSIC
    )

    mtx = np.zeros((4, 4))
    mtx[0:3, 0:3] = R
    mtx[0:3, 3] = np.squeeze(T)
    mtx[-1, -1] = 1

    # this call might cause segmentation fault error. This is due to calling cv2.imshow() and plt.show()
    p3ds = triangulate(imgp1[0], imgp2[0], mtx1, mtx2, R, T)
    p3ds = np.append(p3ds, np.ones((p3ds.shape[0], 1)), axis=-1)
    print(p3ds)
    print((mtx @ p3ds.T).T)

if __name__ == "__main__":
    main()

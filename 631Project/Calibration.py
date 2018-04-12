import numpy as np
import cv2 as cv
import glob


# class Calibration:
#
#     def __init__(self, feed):

        cap = cv.VideoCapture(feed)

        if not cap.isOpened():
            print("Unable to read camera feed")

        # termination criteria

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

        objp = np.zeros((6*7,3), np.float32)

        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.

        objpoints = []  # 3d point in real world space

        imgpoints = []  # 2d points in image plane.

        for i in range(50):

            captured, img = cap.read()
            if not captured:
                break

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners

            ret, corners = cv.findChessboardCorners(gray, (7,6), None)

            # If found, add object points, image points (after refining them)

            if ret:

                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

                imgpoints.append(corners)

                # Draw and display the corners

                cv.drawChessboardCorners(img, (7,6), corners2, ret)

                cv.imshow('img', img)

                cv.waitKey(500)

        # OpenCV Shutdown
        cap.release()

        cv.destroyAllWindows

        ret, self.camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
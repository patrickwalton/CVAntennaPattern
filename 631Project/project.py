import cv2
from Tracker import Tracker
# from Calibration import Calibration
import numpy as np


def main():
    camera_matrix = np.array([[39., 0.,   316.],
                              [0.,  359., 177.],
                              [0.,  0.,   1.]])

    # # Calibrate Camera
    if 'camera_matrix' not in locals():
        camera_matrix = calibrate('OP2Cal.mp4')

    # OpenCV Setup
    cap = cv2.VideoCapture('Sundial.mp4')

    if not cap.isOpened():
        print("Unable to read camera feed")

    if cap.get(3) > 1000:
        frame_width = int(0.33*cap.get(3))
        frame_height = int(0.33*cap.get(4))
    else:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    frame_rate = 30
    frame_interval = 1/frame_rate

    vid = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width, frame_height))

    while cap.isOpened():

        # Camera Capture
        captured, frame = cap.read()

        if not captured:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Tracker Init
        if 'tracker' in locals():
            tracker.update(frame)
        else:
            tracker = Tracker(frame, 20, (frame_width, frame_height), camera_matrix)

        # User Interface
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        cv2.imshow('Frame', tracker.frame)

    # OpenCV Shutdown
    cap.release()
    vid.release()

    cv2.destroyAllWindows()


def calibrate(source):
    # OpenCV Setup
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Unable to read camera feed")

    if cap.get(3) > 1000:
        frame_width = int(0.33 * cap.get(3))
        frame_height = int(0.33 * cap.get(4))
    else:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

    # termination criteria

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

    # checkerboard size
    cbrow = 9
    cbcol = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

    objp = np.zeros((cbrow * cbcol, 3), np.float32)

    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.

    objpoints = []  # 3d point in real world space

    imgpoints = []  # 2d points in image plane.
    #

    while cap.isOpened():
        # Camera Capture
        for i in range(10):
            captured, frame = cap.read()

        if not captured:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners

        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)

        # If found, add object points, image points (after refining them)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners)

            # Draw and display the corners

            cv2.drawChessboardCorners(frame, (cbrow, cbcol), corners2, ret)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # OpenCV Shutdown
    cap.release()

    cv2.destroyAllWindows


    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(camera_matrix)

    return camera_matrix

if __name__ == "__main__":
    main()

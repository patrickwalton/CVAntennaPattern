import cv2
import numpy as np
from Tracker import Tracker
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def main():
    # OpenCV Setup
    cap = cv2.VideoCapture('Morning.mp4')
    frame_width = int(0.33*cap.get(3))
    frame_height = int(0.33*cap.get(4))
    frame_rate = 30
    frame_interval = 1/frame_rate
    f = 0.00435  # focal length, meters
    camera_matrix = np.array([[f, 0, frame_width/2],
                              [0, f, frame_height/2],
                              [0, 0, 1]])

    vid = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width, frame_height))

    if not cap.isOpened():
        print("Unable to read camera feed")

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
            tracker = Tracker(frame, 500, (frame_width, frame_height), camera_matrix)

        # User Interface
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        cv2.imshow('Frame', tracker.frame)

    # OpenCV Shutdown
    cap.release()
    vid.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

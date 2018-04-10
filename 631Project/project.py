import cv2
import numpy as np
from Tracker import Tracker

# OpenCV Setup
cap = cv2.VideoCapture('Morning.mp4')
frame_width = int(0.33*cap.get(3))
frame_height = int(0.33*cap.get(4))
frame_rate = 30
frame_interval = 1/frame_rate

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
        tracker = Tracker(frame, (10, 50), (frame_width, frame_height))

    # User Interface
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    cv2.imshow('Frame', tracker.frame)

# OpenCV Shutdown
cap.release()
vid.release()

cv2.destroyAllWindows()
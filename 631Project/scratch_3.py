import cv2

cap = cv2.VideoCapture(0)

print(cap.isOpened())

if not cap.isOpened():
        print("Unable to read camera feed")

while cap.isOpened():

    # Camera Capture
    captured, frame = cap.read()
    if not captured:
        break

    cv2.imshow('img', frame)
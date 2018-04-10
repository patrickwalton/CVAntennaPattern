import numpy as np
from Point import Point
import cv2


class Tracker:
    # Finds, tracks, and manages points in an image
    def __init__(self, frame, points_range, frame_size):
        self.frame = frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.prior_frame = self.frame
        self.prior_gray_frame = self.gray_frame
        self.frame_size = frame_size

        self.corners = cv2.goodFeaturesToTrack(self.gray_frame,
                                               maxCorners=points_range[1],
                                               qualityLevel=0.3,
                                               minDistance=7,
                                               blockSize=7
                                               )
        self.prior_corners = self.corners

        self.points = []
        for i in range(self.corners.shape[0]):
            self.points.append(Point(tuple(self.corners[i, 0])))

        self.draw()

    def update(self, frame):
        self.step(frame)
        self.corners, self.status, self.error = cv2.calcOpticalFlowPyrLK(self.prior_gray_frame,
                                                                         self.gray_frame,
                                                                         self.prior_corners,
                                                                         None,
                                                                         winSize=(15, 15),
                                                                         maxLevel=2,
                                                                         criteria=(cv2.TERM_CRITERIA_EPS |
                                                                                   cv2.TERM_CRITERIA_COUNT,
                                                                                   10,
                                                                                   0.03)
                                                                         )

        for i in range(self.corners.shape[0]):
            self.points[i].update(tuple(self.corners[i, 0]))

        for point in self.points:
            if point.location[0] < 0 \
                    or point.location[1] < 0 \
                    or point.location[0] > self.frame_size[0] \
                    or point.location[1] > self.frame_size[1]:
                del self.points[self.points.index(point)]
                print('DELETED: ', point.location)

        self.draw()

    def draw(self):
        for point in self.points:
            cv2.line(self.frame, point.prior_location, point.location, (255, 255, 255), 3)
            cv2.circle(self.frame, point.location, 5, (0, 0, 255), -1)

    def step(self, frame):
        self.prior_frame = self.frame
        self.prior_gray_frame = self.gray_frame
        self.frame = frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self.prior_corners = np.zeros((len(self.points), 1, 2), dtype=np.float32)
        for i in range(len(self.points)):
            self.prior_corners[i, 0] = self.points[i].location

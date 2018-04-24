import numpy as np
from Point import Point
from Pose import Pose
import cv2


class Tracker:
    # Finds, tracks, and manages points in an image
    def __init__(self, frame, point_count, frame_size, frame_interval, camera_matrix, frame_count, run_name):
        self.frame = frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        self.prior_frame = self.frame
        self.prior_gray_frame = self.gray_frame
        self.frame_size = frame_size
        self.frame_interval = frame_interval
        self.point_count = point_count
        self.camera_matrix = camera_matrix
        self.i = 0

        self.feature_mask = 255 * np.ones(self.gray_frame.shape).astype(np.uint8)
        w, h = self.feature_mask.shape
        self.feature_mask[int(0.3 * w):int(0.7 * w), int(0.3 * h):int(0.7 * h)] = 0

        self.corners = cv2.goodFeaturesToTrack(self.gray_frame,
                                               maxCorners=self.point_count,
                                               qualityLevel=0.3,
                                               minDistance=7,
                                               blockSize=7,
                                               mask=self.feature_mask
                                               )
        self.prior_corners = self.corners
        self.predicted_corners = self.corners
        self.deficit = 0

        self.points = []
        for i in range(self.corners.shape[0]):
            self.points.append(Point(tuple(self.corners[i, 0]), self.frame_interval))

        self.draw()

        self.pose = Pose(self.camera_matrix, frame_interval, frame_count, run_name)

    def update(self, frame):
        self.step(frame)

        self.corners, _, _ = cv2.calcOpticalFlowPyrLK(self.prior_gray_frame,
                                                      self.gray_frame,
                                                      self.prior_corners,
                                                      # self.predicted_corners,
                                                      None,
                                                      winSize=(15, 15),
                                                      maxLevel=2,
                                                      criteria=(cv2.TERM_CRITERIA_EPS |
                                                                cv2.TERM_CRITERIA_COUNT,
                                                                10,
                                                                0.03),
                                                      # flags=cv2.OPTFLOW_USE_INITIAL_FLOW
                                                      )

        self.pose.update(self.prior_corners, self.corners)

        self.deficit = self.point_count-self.corners.shape[0]

        # if self.i > 5/self.frame_interval:
        #     self.refresh()
        #     self.i = -1
        # elif self.deficit > 0:
        #     self.supplement()
        # self.i += 1

        if self.deficit > 0:
            self.supplement()

        for i in range(self.corners.shape[0]):
            self.points[i].update(tuple(self.corners[i, 0]))

        w, h = self.feature_mask.shape
        for point in self.points:
            if not 0 < point.location[0] < self.frame_size[0]\
                    or not 0 < point.location[1] < self.frame_size[1] \
                    or 0.3*h <= point.location[0] <= 0.7*h \
                    or 0.3*w <= point.location[1] <= 0.7*w:
                del self.points[self.points.index(point)]
                # print('DELETED: ', point.location)

        self.draw()

    def supplement(self):
        corners_supplement = cv2.goodFeaturesToTrack(self.gray_frame,
                                                     maxCorners=self.deficit,
                                                     qualityLevel=0.3,
                                                     minDistance=7,
                                                     blockSize=7,
                                                     mask=self.feature_mask
                                                     )

        for i in range(corners_supplement.shape[0]):
            point = tuple(corners_supplement[i, 0])
            self.points.append(Point(point, self.frame_interval))

        self.corners = np.concatenate((self.corners, corners_supplement), 0)

        self.deficit = 0

    def refresh(self):
        self.corners = cv2.goodFeaturesToTrack(self.gray_frame,
                                               maxCorners=self.point_count,
                                               qualityLevel=0.3,
                                               minDistance=7,
                                               blockSize=7,
                                               mask=self.feature_mask
                                               )
        self.prior_corners = self.corners
        self.predicted_corners = self.corners
        self.deficit = 0

        self.points = []
        for i in range(self.corners.shape[0]):
            self.points.append(Point(tuple(self.corners[i, 0]), self.frame_interval))

    def draw(self):
        for point in self.points:
            cv2.line(self.frame, point.prior_location, point.location, (255, 255, 255), 1)
            cv2.circle(self.frame, point.location, 2, (0, 0, 255), -1)

    def step(self, frame):
        self.prior_frame = self.frame
        self.prior_gray_frame = self.gray_frame
        self.frame = frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self.prior_corners = np.zeros((len(self.points), 1, 2), dtype=np.float32)
        # self.predicted_corners = np.zeros((len(self.points), 1, 2), dtype=np.float32)
        for i in range(len(self.points)):
            self.prior_corners[i, 0] = self.points[i].location
            # self.predicted_corners[i, 0] = self.points[i].predicted_location.T
        self.corners = None

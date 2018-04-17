import cv2
import numpy as np


class Point:
    # Represent frame_interval a point being tracked
    def __init__(self, pixel_coordinates, frame_interval):
        self.location = pixel_coordinates
        self.prior_location = self.location

        # # Set up Point's kalman filter
        # self.kalman = cv2.KalmanFilter(4, 2)
        # self.kalman.transitionMatrix = np.eye(4) + frame_interval * np.eye(4, k=2)
        # self.kalman.measurementMatrix = np.eye(2, 4)
        # self.kalman.processNoiseCov = np.array([[(frame_interval ** 3.) / 3., 0., (frame_interval ** 2.) / 2., 0.],
        #                                         [0., (frame_interval ** 3.) / 3., 0., (frame_interval ** 2.) / 2.],
        #                                         [(frame_interval ** 2.) / 2., 0., frame_interval, 0.],
        #                                         [0., (frame_interval ** 2.) / 2., 0., frame_interval]]
        #                                        ) * 1e+3
        # self.kalman.measurementNoiseCov = 1e+0 * np.eye(2)
        # self.kalman.errorCovPost = 1e+0 * np.eye(4)
        # self.kalman.statePost = np.array([[pixel_coordinates[0]], [pixel_coordinates[1]], [0.], [0.]])
        #
        # # Predict new point
        # self.predicted_location = self.kalman.measurementMatrix.dot(self.kalman.predict())

    def update(self, pixel_coordinates):
        self.prior_location = self.location
        self.location = pixel_coordinates

        # # Correct point using Kalman filter
        # measurement = np.array([[0.], [0.]])
        # measurement[0][0] = self.location[0]
        # measurement[1][0] = self.location[1]
        # # self.location = self.kalman.measurementMatrix.dot(self.kalman.correct(measurement))
        # # self.location = tuple(self.location.reshape(2).astype('float32'))
        #
        # # Predict new point using Kalman filter
        # self.predicted_location = self.kalman.measurementMatrix.dot(self.kalman.predict())

import numpy as np
import cv2
from Power import Power


class Pose:
    # Represents camera pose with respect to initial image in sequence
    #   Includes current and (TODO) historical pose
    def __init__(self, camera_matrix, frame_interval, frame_count, run_name):
        self.camera_matrix = camera_matrix
        self.essential_matrix = np.eye(3)
        self.position = np.array([[0, -20, 0]]).T
        self.orientation = np.array([[1,  0,  0],
                                     [0,  0,  1],
                                     [0,  -1,  0]])
        # self.orientation = np.eye(3)
        self.pose = np.eye(4)
        self.euler = [0, 0, 0]
        self.rot_history = []
        self.step = 0
        self.power = Power(frame_count, run_name)

        self.point_frame = 255 * np.zeros((500, 500, 3))

        self.draw()

        #
        # # Set up Pose's kalman filter
        # self.kalman = cv2.KalmanFilter(2, 1)
        # self.kalman.transitionMatrix = np.eye(2) + frame_interval * np.eye(2, k=1)
        # self.kalman.measurementMatrix = np.eye(1, 2)
        # self.kalman.processNoiseCov = np.array([[(frame_interval ** 3.) / 3., 0.],
        #                                         [0., (frame_interval ** 3.) / 3.]]
        #                                        ) * 1e-3 # Book: 1e-5
        # self.kalman.measurementNoiseCov = 0.1 * np.ones((1, 1))- self.power.peak_power[self.step]
        # self.kalman.errorCovPost = 1. * np.ones((2, 2))  # Needs to be identity at first to maintain meaningfulness,updated later
        # self.kalman.statePost = np.array([[0.], [0.]])
        # self.kalman.predict()

    def update(self, prior_corners, corners):
        self.essential_matrix, mask = cv2.findEssentialMat(prior_corners,
                                                           corners,
                                                           self.camera_matrix,
                                                           method=cv2.RANSAC,
                                                           prob=0.999,
                                                           threshold=0.1
                                                           )

        retval, R, t, mask2 = cv2.recoverPose(self.essential_matrix,
                                              prior_corners,
                                              corners,
                                              self.camera_matrix,
                                              mask
                                              )

        # H, mask = cv2.findHomography(prior_corners, corners, method=0, ransacReprojThreshold=3)
        #
        # retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)
        #
        # for i in range(4):
        #     # if translations[i][2] > 0:
        #     projection_matrix_1 = np.matmul(self.camera_matrix, np.concatenate((np.eye(3), 0.*t), 1))
        #     projection_matrix_2 = np.matmul(self.camera_matrix, np.concatenate((R, t), 1))
        #     test_points = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, prior_corners, corners)
        #     # print(test_points)
        #     # test_point = np.matmul(np.matmul(rotations[i], ), test_point)
        #     print('i', i)
        #     # print(test_points)
        #     print(np.less(test_points[2], 0))
        #     # for j in range(test_points.shape[0]):
        #     #     test_point = test_points[j]/test_points[j, 3]
        #     #     print('j', j)
        #     #     # print(test_point)
        #     #     if test_point[2] < 0:
        #     #         # R = rotations[i]
        #     #         break

        # R, R2, t = cv2.decomposeEssentialMat(self.essential_matrix)
        #
        # if np.trace(R) < 2.5:
        #     R = R2
        # print(np.trace(R))

        self.orientation = np.matmul(R, self.orientation)

        self.position = self.position - np.dot(self.orientation.T, t)

        self.pose = np.concatenate((np.concatenate((self.orientation.T,
                                                   # self.position), 1),
                                                    np.zeros((3, 1))), 1),
                                   np.array([[0, 0, 0, 1]])), 0)

        self.draw()

        # # Correct point using Kalman filter
        # measurement = np.array([[0.]]).astype('float64')
        # measurement[0][0] = self.euler[1]
        # self.rot_history.append(np.matmul(self.kalman.measurementMatrix, self.kalman.correct(measurement))[0, 0])
        # self.kalman.predict()

    def draw(self):

        # Square location and size parameters
        c, a = 0, 50

        # Initialize Square
        origin = np.array([[c, a, c, c],
                           [c, c, a, c],
                           [c, c, c, a],
                           [1, 1, 1, 1]])

        # Transform Square
        origin = np.matmul(self.pose, origin) + 250 * np.ones((4, 4))

        # Initialize Plot Frame
        plot_frame = 255 * np.ones((500, 500, 3))

        # World N-S Axis (+x is N)
        cv2.line(plot_frame, (200, 250), (300, 250), (0, 0, 0), 1 )
        cv2.putText(plot_frame, 'N', (300, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(plot_frame, 'S', (200, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # World E-W Axis (+y is E)
        cv2.line(plot_frame, (250, 200), (250, 300), (0, 0, 0), 1)
        cv2.putText(plot_frame, 'W', (250, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(plot_frame, 'E', (250, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # dB Axes
        max_power = np.max(self.power.peak_power)
        cv2.circle(plot_frame, (250, 250), 200, (0, 0, 0), 1)
        cv2.putText(plot_frame, 'Peak', (240, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        cv2.circle(plot_frame, (250, 250), 150, (0, 0, 0), 1)
        cv2.putText(plot_frame, '-5dB', (240, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        cv2.circle(plot_frame, (250, 250), 100, (0, 0, 0), 1)
        cv2.putText(plot_frame, '-10dB', (240, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # Add Transformed Square to Plot
        for i in range(origin.shape[1]-1):
            color = [0, 0, 0]
            color[2-i] = 255
            cv2.line(plot_frame,
                     tuple(origin[0:2, 0].astype(np.int)),
                     tuple(origin[0:2, i+1].astype(np.int)),
                     tuple(color),
                     3
                     )

        point = origin[0:2, 1] - 250 * np.ones((1, 2))
        point = (200 + 10*self.power.peak_power[self.step])*(point/a) + 250 * np.ones((1, 2))
        self.step += 1

        cv2.circle(self.point_frame, tuple(point.astype(int).reshape(2)), 3, (255, 255, 255), -1)

        plot_frame -= self.point_frame

        # Plot Transformed Square
        cv2.imshow('square', plot_frame)

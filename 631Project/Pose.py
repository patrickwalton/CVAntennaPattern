import numpy as np
import cv2

class Pose:
    # Represents camera pose with respect to initial image in sequence
    #   Includes current and historical pose
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.essential_matrix = np.eye(3)
        self.position = np.zeros((3, 1))
        self.orientation = np.eye(3)
        self.pose = np.eye(4)
        self.orientationh = np.eye(4)

        self.draw()

    def update(self, prior_corners, corners):
        self.essential_matrix, mask = cv2.findEssentialMat(prior_corners,
                                                           corners,
                                                           self.camera_matrix,
                                                           method=0
                                                           )

        retval, R, t, mask2 = cv2.recoverPose(self.essential_matrix,
                                              prior_corners,
                                              corners,
                                              self.camera_matrix,
                                              mask
                                              )

        # R1, R2, t = cv2.decomposeEssentialMat(self.essential_matrix)
        #
        # if np.trace(R1) < 2.5:
        #     R = R1
        # else:
        #     R = R2

        self.orientation = np.matmul(R, self.orientation)

        self.position = self.position + np.dot(R, t)

        self.orientationh = np.concatenate((np.concatenate((self.orientation,
                                                   np.zeros((3, 1))), 1),
                                   np.ones((1, 4))), 0)

        self.pose = np.concatenate((np.concatenate((self.orientation,
                                                   self.position), 1),
                                   np.ones((1, 4))), 0)

        # print(np.concatenate((self.orientation, self.position), 1))

        self.draw()

    def draw(self):
        cx, cy, a = 30, 30, 10

        square = np.array([[cx+a, cx-a, cx-a, cx+a, cx+a],
                           [cy+a, cy+a, cy-a, cy-a, cy+a],
                           [1,    1,    1,    1,    1],
                           [1,    1,    1,    1,    1]])

        square2 = np.matmul(self.orientationh, square) + 250 * np.ones((4, 5))

        square = np.matmul(self.pose, square) + 250 * np.ones((4, 5))

        plot_frame = 255 * np.ones((500, 500))

        cv2.line(plot_frame,
                 (200, 250),
                 (300, 250),
                 (0, 0, 0),
                 1
                 )

        cv2.line(plot_frame,
                 (250, 200),
                 (250, 300),
                 (0, 0, 0),
                 1
                 )

        for i in range(square.shape[1]-1):
            cv2.line(plot_frame,
                     tuple(square[0:2, i].astype(np.int)),
                     tuple(square[0:2, i+1].astype(np.int)),
                     (0, 0, 0),
                     3
                     )

        for i in range(square2.shape[1]-1):
            cv2.line(plot_frame,
                     tuple(square2[0:2, i].astype(np.int)),
                     tuple(square2[0:2, i+1].astype(np.int)),
                     (0, 0, 0),
                     1
                     )

        cv2.imshow('square', plot_frame)

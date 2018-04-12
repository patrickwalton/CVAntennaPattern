import numpy as np
import cv2


class Pose:
    # Represents camera pose with respect to initial image in sequence
    #   Includes current and (TODO) historical pose
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.essential_matrix = np.eye(3)
        self.position = np.array([[0, -2, 0]]).T
        self.orientation = np.array([[1,  0,  0],
                                     [0,  0,  1],
                                     [0,  -1,  0]])
        # self.orientation = np.eye(3)
        self.pose = np.eye(4)
        self.euler = [0, 0, 0]
        self.rot_history = []

        self.draw()

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

        self.orientation = np.matmul(R, self.orientation)

        self.rotation_check()

        self.position = self.position + np.dot(self.orientation, t)

        self.pose = np.concatenate((np.concatenate((self.orientation.T,
                                                   # self.position), 1),
                                                    np.zeros((3, 1))), 1),
                                   np.array([[0, 0, 0, 1]])), 0)

        self.draw()

        self.rot2eul()

        self.rot_history.append(self.euler)

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

        # self.rot_history.append(np.arctan2(origin[1, 1], origin[0, 1]))

        # Initialize Plot Frame
        plot_frame = 255 * np.ones((500, 500, 3))

        # Add Coordinate Axis
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

        # Plot Transformed Square
        cv2.imshow('square', plot_frame)

        # cv2.waitKey(0)

    def rotation_check(self):
        # Checks if a matrix is a valid rotation matrix.
        diff = np.linalg.norm(np.identity(3, dtype=self.orientation.dtype)
                              - np.dot(self.orientation.T, self.orientation))
        if diff > 1e-6:
            print('Invalid Rotation: R.T*R != I')

        diff = 1 - np.linalg.det(self.orientation)
        if diff > 1e-6:
            print('Invalid Rotation: det(R) = ', np.linalg.det(self.orientation))

    def rot2eul(self):
        R = self.orientation
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        self.euler = [x, y, z]

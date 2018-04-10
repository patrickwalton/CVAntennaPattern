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

    def update(self, prior_corners, corners):
        self.essential_matrix, mask = cv2.findEssentialMat(prior_corners,
                                                           corners,
                                                           self.camera_matrix,
                                                           method=cv2.RANSAC,
                                                           prob=0.999,
                                                           threshold=1.0
                                                           )

        retval, R, t, mask2 = cv2.recoverPose(self.essential_matrix,
                                              prior_corners,
                                              corners,
                                              self.camera_matrix
                                              )

        self.orientation = np.matmul(R, self.orientation)

        self.position = self.position + t

        self.rotationMatrixToEulerAngles()
        print(self.x, self.y, self.z)

        # print(t, '/n', self.position)

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self):
        shouldBeIdentity = np.dot(self.orientation.T, self.orientation)
        I = np.identity(3, dtype=self.orientation.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self):

        assert (self.isRotationMatrix())

        sy = np.sqrt(self.orientation[0, 0] * self.orientation[0, 0] + self.orientation[1, 0] * self.orientation[1, 0])

        singular = sy < 1e-6

        if not singular:
            self.x = np.arctan2(self.orientation[2, 1], self.orientation[2, 2])
            self.y = np.arctan2(-self.orientation[2, 0], sy)
            self.z = np.arctan2(self.orientation[1, 0], self.orientation[0, 0])
        else:
            self.x = np.arctan2(-self.orientation[1, 2], self.orientation[1, 1])
            self.y = np.arctan2(-self.orientation[2, 0], sy)
            self.z = 0


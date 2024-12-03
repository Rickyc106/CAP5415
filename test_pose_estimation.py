#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from test_feature_matching import FeatureMatcher

class PoseEstimation(object):
    def __init__(self):
        # Camera intrinsic matrix, TUM Freiburg2
        self.K = np.array([[520.9, 0., 325.1],
                           [0., 521., 249.7],
                           [0., 0., 1.]])

    def estimate(self, kp0, kp1, matches):
        # Select only matched keypoints
        p0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])

        # Calculate essential matrix
        E, _ = cv2.findEssentialMat(p0,
                                    p1,
                                    self.K,
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=1.0)
        
        # Recover rotation / translation from essential matrix
        _, R, t, mask = cv2.recoverPose(E, p0, p1, self.K)
        print('Rotation:', R)
        print('Translation:', t)
        print('Essential:', E)

        # Check E = t^R*scale
        t_x = np.array([[0., -t[2,0], t[1,0]],
                        [t[2,0], 0., -t[0,0]],
                        [-t[1,0], t[0,0], 0.]])
        print('Epipolar Constraint:', t_x @ R)

        # Triangulate points in 3-D space
        T1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        T2 = np.hstack((R, t))
        P1 = np.dot(self.K,  T1)
        P2 = np.dot(self.K,  T2)
        point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(p0, axis=1), np.expand_dims(p1, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        return point_3d
    
    def display_triangulated_points(self, point_3d):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(list(map(lambda x: x[0], point_3d)),
                   list(map(lambda x: x[1], point_3d)),
                   list(map(lambda x: x[2], point_3d)))
        print(np.mean(point_3d, axis=0))
        plt.show()
    
if __name__ == '__main__':
    # Parse input arguments
    argparser = argparse.ArgumentParser(description='Performs pose estimation on two sample images.')
    args = argparser.parse_args()

    # Read sample images
    img1 = cv2.imread('./test-images/1.png')
    img2 = cv2.imread('./test-images/2.png')

    # Perform feature matching algorithm
    FM = FeatureMatcher()
    kp0, kp1, matches = FM.match(img1, img2)

    # Perform pose estimation
    PE = PoseEstimation()
    point_3d = PE.estimate(kp0, kp1, matches)

    # Display results
    PE.display_triangulated_points(point_3d)
    FM.display_matches(img1, img2, kp0, kp1, matches)
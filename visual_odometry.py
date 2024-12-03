#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

class VisualOdometry(object):
    def __init__(self):
        self.detector = cv2.FastFeatureDetector_create()
        self.p0 = None
        self.p1 = None
        self.d0 = None
        self.d1 = None
        self.st_params = dict(maxCorners=100,
                              qualityLevel=0.01,
                              minDistance=10)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.colors = np.random.randint(0, 255, (100, 3))
        self.prev_frame = None
        self.curr_frame = None
        self.FOV = 90.0
        self.focal_length = 1000
        self.principal_point = (0, 0)
        self.R = None
        self.t = None
        self.intrinsic_matrix = np.array([[400, 0, 300],
                                          [0, 400, 400],
                                          [0, 0, 1]])
        self.P = np.concatenate((self.intrinsic_matrix, np.zeros((3,1))), axis=1)
        self.rotation = np.eye(3)
        self.position = np.zeros((3,1))
        self.write_to_log = True
        self.filename = './data/estimate-4.log'
        self.num_of_features = 0
        self.pose = np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), [0, 0, 0, 1]))

    def detect_features(self, frame):
        '''
        frame: Single frame from video source
        '''
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints / descriptors
        orb = cv2.ORB_create(3000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors

    def match_features_brute_force(self, kp0, kp1, d0, d1):
        '''
        kp0: Previous keypoints from ORB detection
        kp1: Current keypoints
        d0: Previous descriptors from ORB detection
        d1: Current descriptors
        '''
        # Use brute-force matching with hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d0, d1)
        matches = sorted(matches, key=lambda x: x.distance)

        img_with_matches = cv2.drawMatches(self.prev_frame, kp0,
                                           self.curr_frame, kp1,
                                           matches[:50],
                                           None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('frame', img_with_matches)

        # Return matched features only
        p0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        p1 = np.float32([kp1[m.trainIdx].pt for m in matches])

        return p0, p1
    
    def match_features_flann(self, kp0, kp1, d0, d1):
        '''
        kp0: Previous keypoints from ORB detection
        kp1: Current keypoints
        d0: Previous descriptors from ORB detection
        d1: Current descriptors
        '''
        # Use FLANN-based matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(d0, d1, k=2)

        # Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
        img_with_matches = cv2.drawMatches(self.prev_frame, kp0,
                                           self.curr_frame, kp1,
                                           good, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('image', img_with_matches)
        # cv2.waitKey(200)

        # Return features with good matches per ratio test
        p0 = np.float32([kp0[m.queryIdx].pt for m in good])
        p1 = np.float32([kp1[m.trainIdx].pt for m in good])

        return p0, p1
    
    def estimate_ego_motion(self, keypoints, matches):
        '''
        keypoints: Keypoints from feature extraction
        matches: Feature matches between subsequent frames
        '''
        # Select only matched keypoints
        p0 = np.float32([self.p0[m.queryIdx].pt for m in matches])
        p1 = np.float32([keypoints[m.queryIdx].pt for m in matches])

        # Estimate the essential matrix
        E, _ = cv2.findEssentialMat(p0,
                                    p1,
                                    self.intrinsic_matrix,
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=1.0)
        
        # Recover relative pose from essential matrix
        _, R, t, _ = cv2.recoverPose(E, p0, p1, self.intrinsic_matrix)
        
        return R, t
    
    def compute_3d_point_triangulation(self, R, t, p0, p1):
        T1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        T2 = np.hstack((R, t))
        P1 = np.dot(self.intrinsic_matrix,  T1)
        P2 = np.dot(self.intrinsic_matrix,  T2)
        point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(p0, axis=1), np.expand_dims(p1, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        return point_3d
    
    def compute_mean_translation(self, point_3d):
        return np.mean(point_3d, axis=0)
    
    def create_tf(self, R, t):
        tf = np.eye(4, dtype=np.float32)
        tf[:3, :3] = R
        tf[:3, 3] = t
        return tf
    
    def compute_pose(self, p0, p1):
        # Estimate essential matrix of transformation between subsequent frames
        E, _ = cv2.findEssentialMat(p0, p1,
                                    self.intrinsic_matrix,
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=1.0)
        
        # Decompose essential matrix into R and t
        _, R, t, _ = cv2.recoverPose(E, p0, p1, self.intrinsic_matrix)

        # Get transformation matrix
        tf = self.create_tf(R, np.squeeze(t))
        return tf

    def decompose_essential_mat(self, E, p0, p1):
        def sum_z_cal_relative_scale(R, t):
            # T1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            # T2 = np.hstack((R, t.reshape((3,1))))
            # P1 = np.dot(self.intrinsic_matrix, T1)
            # P2 = np.dot(self.intrinsic_matrix, T2)
            # point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(p0, axis=1), np.expand_dims(p1, axis=1))
            # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))

            # Get the transformation matrix
            T = self.create_tf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.intrinsic_matrix, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, p0.T, p1.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
        
        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]
    
    def process_frame_3(self, frame):
        # Initial frame won't have a previous frame yet
        if self.prev_frame is None:
            self.prev_frame = frame
            # Write initial pose to log file
            if self.write_to_log:
                with open(self.filename, 'w') as outfile:
                    mat = np.hstack((self.rotation, self.position))
                    np.savetxt(outfile, mat, fmt='%-7.2f')
        # Second frame won't have a current frame yet
        elif self.curr_frame is None:
            self.curr_frame = frame
            self.perform_visual_odometry_3()
        # All other frames can use a sliding window approach
        else:
            self.prev_frame = self.curr_frame
            self.curr_frame = frame
            self.perform_visual_odometry_3()

    def perform_visual_odometry_3(self):
        kp0, d0 = self.detect_features(self.prev_frame)
        kp1, d1 = self.detect_features(self.curr_frame)
        p0, p1 = self.match_features_brute_force(kp0, kp1, d0, d1)
        transformation_matrix = self.compute_pose(p0, p1)
        self.pose = self.pose @ np.linalg.inv(transformation_matrix)
        return self.pose
    
    def run_3(self, cap):
        if not cap.isOpened():
            raise('Error reading video file')
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            count += 1

            # Loop back to beginning if reached last frame
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                count = 0
                self.write_to_log = False
                continue

            # Process each frame one at a time
            self.process_frame_3(frame)

            if self.curr_frame is None:
                continue

            self.rotation = self.pose[:3, :3]
            self.position = self.pose[:3, 3].reshape((3,1))

            assert self.rotation.shape == (3,3)
            assert self.position.shape == (3,1)

            # Write initial pose to log file
            if self.write_to_log:
                with open(self.filename, 'a') as outfile:
                    mat = np.hstack((self.rotation, self.position))
                    np.savetxt(outfile, mat, fmt='%-7.2f')

            # OpenCV window control
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                # Wait until another key is pressed
                cv2.waitKey(-1)
    
    def process_frame(self, frame):
        # Initial frame won't have a previous frame yet
        if self.prev_frame is None:
            self.prev_frame = frame
            # Write initial pose to log file
            if self.write_to_log:
                with open(self.filename, 'w') as outfile:
                    mat = np.hstack((self.rotation, self.position))
                    np.savetxt(outfile, mat, fmt='%-7.2f')
        # Second frame won't have a current frame yet
        elif self.curr_frame is None:
            self.curr_frame = frame
            self.perform_visual_odometry()
        # All other frames can use a sliding window approach
        else:
            self.prev_frame = self.curr_frame
            self.curr_frame = frame
            self.perform_visual_odometry()

    def perform_visual_odometry(self):
        # Perform feature detection once features are no longer visible in frame
        if self.num_of_features < 250:
            print('Features below threshold! Detecting new features...')
            self.p0, self.d0 = self.detect_features(self.prev_frame)
            # Convert ORB keypoints into numpy arrays
            self.p0 = np.float32([x.pt for x in self.p0]).reshape(-1, 1, 2)

        # Calculate optical flow using Lucas-Kanade method
        self.p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame,
                                                      self.curr_frame,
                                                      self.p0,
                                                      None,
                                                      **self.lk_params)

        # Select good matches
        if self.p1 is not None:
            good_old = self.p0[status==1]
            good_new = self.p1[status==1]

        # Estimate essential matrix of transformation between subsequent frames
        E, _ = cv2.findEssentialMat(good_old, good_new,
                                    self.intrinsic_matrix,
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=1.0)
        
        # Recover relative pose from essential matrix
        _, R, t, _ = cv2.recoverPose(E, good_old, good_new, self.intrinsic_matrix)

        # Update rotation / translation matrices
        scale = 1.0
        if abs(t[2,0]) > abs(t[0,0]) and abs(t[2,0]) > abs(t[1,0]):
            self.rotation = self.rotation @ R.T
            self.position = self.position + scale * R.T @ t

        # Write each updated pose to output file
        if self.write_to_log:
            with open(self.filename, 'a') as outfile:
                mat = np.hstack((self.rotation, self.position))
                np.savetxt(outfile, mat, fmt='%-7.2f')

        # Reset number of detected features
        self.num_of_features = good_new.shape[0]

    def run_new(self, cap):
        if not cap.isOpened():
            raise('Error reading video file')
        
        while cap.isOpened():
            ret, frame = cap.read()

            # Loop back to beginning if reached last frame
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.write_to_log = False
                continue

            # Process each frame one at a time
            self.process_frame(frame)

            if self.curr_frame is None:
                continue

            # Compute brute-force matches for visualization purposes only
            p0, d0 = self.detect_features(self.prev_frame)
            p1, d1 = self.detect_features(self.curr_frame)
            matches = self.match_features(d0, d1)

            # Display consecutive frames with matches
            img_with_matches = cv2.drawMatches(self.prev_frame, p0,
                                               self.curr_frame, p1,
                                               matches[:50],
                                               None,
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('frame', img_with_matches)

            # OpenCV window control
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                # Wait until another key is pressed
                cv2.waitKey(-1)

    def run(self, cap):
        if not cap.isOpened():
            raise('Error reading video file')

        count = 0

        write_to_log = True
        while cap.isOpened():
            ret, frame = cap.read()

            # Loop back to beginning if reached last frame
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                write_to_log = False
                continue
            
            keypoints, descriptors = self.extract_features(frame)

            # Initial frame won't have a previous frame yet
            if self.prev_frame is None:
                self.prev_frame = frame
                self.p0 = keypoints
                self.d0 = descriptors
                if write_to_log:
                    filename = './data/estimate-4.log'
                    with open(filename, 'a') as outfile:
                        mat = np.concatenate((self.rotation, self.position), axis=1)
                        np.savetxt(outfile, mat, fmt='%-7.2f')
                continue

            # # Calculate optical flow using Lucas-Kanade method
            # self.p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame,
            #                                               frame,
            #                                               np.float32([kp.pt for kp in self.p0]).reshape(-1, 1, 2),
            #                                               None,
            #                                               **self.lk_params)

            # # Select good matches
            # if self.p1 is not None:
            #     good_new = self.p1[status==1]
            #     print(f'Frame {count}: len(kp): {len(keypoints)} len(good_new): {len(good_new)}')

            # Perform visual odometry algorithm
            matches = self.match_features(self.d0, descriptors)
            R, t, p0, p1 = self.estimate_ego_motion(keypoints, matches)
            # point_3d = self.compute_3d_point_triangulation(R, t, p0, p1)

            # Convert OpenCV camera coordinate system into UE4 coordinate system
            #   . z                   ^ z
            #  /                      |
            # +-------> x    to:      |
            # |                       | . x
            # |                       |/
            # v y                     +-------> y
            # t = np.array([[0, 0, 1],
            #               [1, 0, 0],
            #               [0, -1, 0]]) @ t
            # t = np.array([t[2], t[0], t[1] * -1])

            # Compute relative pose updates
            # self.rotation = self.rotation @ R.T
            # self.position = self.position - (1) * R.T @ t
            
            scale = 1.0
            if abs(t[2,0]) > abs(t[0,0]) and abs(t[2,0]) > abs(t[1,0]):
                self.rotation = R @ self.rotation
                self.position = self.position + scale * R @ t
            # print(f'Frame: {count} -- Position: {self.position.flatten()}')

            # # Update previous variables with new features
            # self.prev_frame = frame
            # self.p0 = keypoints
            # self.d0 = descriptors

            if write_to_log:
                filename = './data/estimate-4.log'
                with open(filename, 'a') as outfile:
                    mat = np.concatenate((self.rotation, self.position), axis=1)
                    np.savetxt(outfile, mat, fmt='%-7.2f')

            # print(f'Frame {count}: {self.position.flatten()}')
            count += 1

            # Display modified frame
            img_with_matches = cv2.drawMatches(self.prev_frame,
                                               self.p0,
                                               frame,
                                               keypoints,
                                               matches[:50],
                                               None,
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('frame', img_with_matches)

            # OpenCV window control
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                # Wait until another key is pressed
                cv2.waitKey(-1)

    # def run(self, cap):
    #     if not cap.isOpened():
    #         raise('Error reading video file')
        
    #     count = 0
    #     while cap.isOpened():
    #         ret, frame = cap.read()

    #         # # Calculate focal length of camera
    #         # if self.focal_length is None:
    #         #     width = frame.shape[1]
    #         #     # self.focal_length = width / (2. * np.tan(self.FOV * np.pi / 360.))
    #         #     self.focal_length = 1000.

    #         # # Set principal point of camera
    #         # if self.principal_point is None:
    #         #     height = frame.shape[0]
    #         #     width = frame.shape[1]
    #         #     # self.principal_point = (width/2, height/2)
    #         #     self.principal_point = (0., 0.)

    #         # Loop back to beginning if reached last frame
    #         if not ret:
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #             continue

    #         # # Convert frame to grayscale
    #         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #         # # Continue to next frame if reading first frame
    #         # if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
    #         #     self.prev_frame = gray
    #         #     # self.p0 = self.detect_features(frame)
    #         #     self.p0 = cv2.goodFeaturesToTrack(gray, **self.st_params)
    #         #     mask = np.zeros_like(frame)
    #         #     continue

    #         # # Calculate optical flow using Lucas-Kanade method
    #         # self.p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame,
    #         #                                           gray,
    #         #                                           self.p0,
    #         #                                           None,
    #         #                                           **self.lk_params)
            
    #         # # Select good matches
    #         # if self.p1 is not None:
    #         #     good_new = self.p1[st==1]
    #         #     good_old = self.p0[st==1]

    #         # # Compute essential matrix that relates consecutive frames
    #         # E, _ = cv2.findEssentialMat(good_new,
    #         #                             good_old,
    #         #                             self.focal_length,
    #         #                             self.principal_point,
    #         #                             cv2.RANSAC,
    #         #                             prob=0.999,
    #         #                             threshold=1.0,
    #         #                             maxIters=1000,
    #         #                             mask=None)
            
    #         # # Recover pose from essential matrix, E
    #         # _, R, t, _ = cv2.recoverPose(E,
    #         #                              good_old,
    #         #                              good_new,
    #         #                              focal=self.focal_length,
    #         #                              pp=self.principal_point,
    #         #                              mask=None)

    #         # # Set initial pose
    #         # if self.R is None:
    #         #     self.R = R
    #         #     self.t = t
    #         # # Calculate pose relative to initial pose
    #         # else:
    #         #     self.R = R @ self.R
    #         #     self.t = self.t + self.R @ t

    #         # if count % 10 == 0:
    #         #     print(f'==== Frame {count} ====')
    #         #     print(self.R)
    #         #     print(self.t)

    #         # count += 1

    #         # # Draw tracks
    #         # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #         #     a, b = new.ravel().astype(np.int32)
    #         #     c, d = old.ravel().astype(np.int32)
    #         #     mask = cv2.line(img=mask, pt1=(a,b), pt2=(c,d),
    #         #                     color=self.colors[i % 100].tolist(), thickness=2)
    #         #     frame = cv2.circle(img=frame, center=(a,b), radius=5,
    #         #                     color=self.colors[i % 100].tolist(), thickness=-1)

    #         # Display original frame
    #         # cv2.imshow('frame', frame)

    #         # # Display modified frame
    #         # img = cv2.add(frame, mask)
    #         # cv2.imshow('frame', img)
    #         # if cv2.waitKey(1000) == ord('q'):
    #         #     break

    # def detect_features(self, frame):
    #     keypoints = self.detector.detect(frame, None)
    #     return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

if __name__ == '__main__':
    # Parse input arguments
    argparser = argparse.ArgumentParser(description='Performs visual odometry estimation on *.avi files.')
    argparser.add_argument(
        '--index',
        default=0,
        type=int,
        help='Index number (default: 0)')
    argparser.add_argument(
        '--path',
        default='./data',
        type=str,
        help='File path to *.avi file recordings')
    args = argparser.parse_args()

    # Perform visual odometry sequence
    try:
        cap = cv2.VideoCapture(f'data/output-{args.index}.avi')
        VO = VisualOdometry()
        VO.run_3(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
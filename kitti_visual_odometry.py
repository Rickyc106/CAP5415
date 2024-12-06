#!/usr/bin/env python3

import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# -- VisualOdometry ------------------------------------------------------------
# ==============================================================================

class VisualOdometry(object):
    def __init__(self, args):
        # Load input data
        self.images = self.load_images(args.dataset, args.nimages)
        self.K = self.load_calibration_params(args.dataset)
        self.ground_truth_poses = self.load_ground_truth_poses(args.dataset, args.nimages)
        assert len(self.images) > 0
        assert len(self.images) == len(self.ground_truth_poses)
        assert self.K.shape == (3,3)

        # VO algorithm initial data members
        self.prev_frame = None
        self.curr_frame = None
        # Pose is expressed in homogeneous coordinates, i.e. 4x4 instead of 3x4
        self.homogenous_pose = np.eye(4)
        # But when comparing against grouth truth, reshape back into 3x4 [R|t] matrices
        self.estimated_poses = [self.homogenous_pose[:3,:4]]

        # Configure VO pipeline
        if args.method == 'matching':
            # Configure feature detectors
            if args.feature == 'SIFT':
                self.detect_method = self.detect_features_sift
                self.flann_index = 0 # FLANN_INDEX_KDTREE
                self.bf_norm_type = cv2.NORM_L2
                self.bf_cross_check = False
                self.nfeatures = args.nfeatures
                print(f'Creating SIFT detector with {self.nfeatures} features')
            elif args.feature == 'ORB':
                self.detect_method = self.detect_features_orb
                self.flann_index = 6 # FLANN_INDEX_LSH
                self.bf_norm_type = cv2.NORM_HAMMING
                self.bf_cross_check = True
                self.nfeatures = args.nfeatures
                print(f'Creating ORB detector with {self.nfeatures} features')
            else:
                raise AssertionError(f'Cannot use {args.feature} with matching-based visual odometry')
            # Configure feature matchers
            if args.matcher == 'BF':
                self.match_method = self.match_features_brute_force
                if args.threshold is not None: print('Warning: BF matcher does not use threshold value')
            elif args.matcher == 'FLANN':
                self.match_method = self.match_features_flann
                self.threshold = args.threshold
        # Configure using optical flow method
        elif args.method == 'optical_flow':
            # ORB / SIFT aren't configured to perform optical-flow tracking, only feature matching
            assert args.feature == 'FAST', f'Cannot use {args.feature} with optical-flow tracking'
            self.detect_method = self.detect_features_fast
            self.match_method = None

        # Logging
        self.write_to_log = False

    @staticmethod
    def load_images(dataset, nimages):
        '''
        This is a helper method for loading input images from a specified dataset.

        Args:
            dataset: Path to KITTI dataset
            nimages: Max number of images to load
        '''
        images = []
        images_path = os.path.join(dataset, 'image_0')
        for i, filename in enumerate(sorted(os.listdir(images_path))):
            if i >= nimages:
                break
            f = os.path.join(images_path, filename)
            if os.path.isfile(f):
                images.append(cv2.imread(f))
        print(f'Loaded {len(images)} image files from dataset: {dataset}')
        return images
    
    @staticmethod
    def load_calibration_params(dataset):
        '''
        This is a helper method for loading calibration parameters from a specified dataset.

        Args:
            dataset: Path to KITTI dataset
        '''
        calibration_path = os.path.join(dataset, 'calib.txt')
        calibration = np.loadtxt(calibration_path)
        # KITTI embeds intrinsic and extrinsic parameters inside two projection matrices,
        # one for each stereo camera. Since we're only dealing with monocular visual odometry,
        # we only need to pull intrinsic parameters for the left stereo camera.
        K = calibration[0].reshape(3,4)[:3,:3]
        return K
    
    @staticmethod
    def load_ground_truth_poses(dataset, nimages):
        '''
        This is a helper method for loading ground truth poses from a specified dataset.

        Args:
            dataset: Path to KITTI dataset
            nimages: Max number of ground truth poses to load
        '''
        ground_truth_path = os.path.join(dataset, 'poses.txt')
        poses = np.loadtxt(ground_truth_path)
        # Ground truth poses are stored as "stacked" 3x4 matrices: [R|t],
        # where R is rotation, t is translation. We use -1 to let numpy
        # figure out how many total poses there are on its own.
        return poses.reshape(-1,3,4)[:nimages]

    def detect_features_harris(self, frame):
        '''
        This method performs Harris corner detection.

        Args:
            frame: Single frame from video source
        '''
        # Detect Harris corners
        dst = cv2.cornerHarris(frame,
                               blockSize=4,
                               ksize=3,
                               k=0.04)
        dst = cv2.dilate(dst, None)
        _, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)

        # Find centroids
        _, _, _, centroids = cv2.connectedComponentsWithStats(dst)

        # Refine corners with sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(frame, np.float32(centroids),
                                   winSize=(5,5), zeroZone=(-1, -1),
                                   criteria=criteria)
        
        return corners
    
    def detect_features_shi_tomasi(self, frame):
        '''
        This method performs Shi-Tomasi corner detection.

        Args:
            frame: Single frame from video source
        '''
        # Detect Shi-Tomasi corners
        corners = cv2.goodFeaturesToTrack(frame,
                                          mask=None,
                                          maxCorners=100,
                                          qualityLevel=0.01,
                                          minDistance=10)
        
        return corners
    
    def detect_features_fast(self, frame):
        '''
        This method performs FAST feature detection.

        Args:
            frame: Single frame from video source
        '''
        # Detect FAST keypoints
        fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        keypoints = fast.detect(frame, None)
        keypoints = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        
        return keypoints
    
    def detect_features_brief(self, frame):
        '''
        This method performs BRIEF feature detection.

        Args:
            frame: Single frame from video source
        '''
        # Detect FAST keypoints / BRIEF descriptors
        fast = cv2.FastFeatureDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        keypoints = fast.detect(frame, None)
        keypoints, descriptors = brief.compute(frame, keypoints)

        return keypoints, descriptors
    
    def detect_features_sift(self, frame):
        '''
        This method performs SIFT feature detection.

        Args:
            frame: Single frame from video source
        '''
        # Detect SIFT keypoints / descriptors
        sift = cv2.SIFT_create(self.nfeatures)
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        
        return keypoints, descriptors

    def detect_features_orb(self, frame):
        '''
        This method performs ORB feature detection.

        Args:
            frame: Single frame from video source
        '''
        # Detect ORB keypoints / descriptors
        orb = cv2.ORB_create(self.nfeatures)
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        
        return keypoints, descriptors

    def match_features_brute_force(self, kp0, kp1, des0, des1, draw_matches=True):
        '''
        This method matches feature descriptors using brute-force method.

        Args:
            kp0: Previous keypoints
            kp1: Current keypoints
            des0: Previous descriptors
            des1: Current descriptors
            draw_matches: Whether to display feature matches
        '''
        # Use brute-force matching
        bf = cv2.BFMatcher(self.bf_norm_type, crossCheck=self.bf_cross_check)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)

        if draw_matches:
            img_with_matches = cv2.drawMatches(self.prev_frame, kp0,
                                               self.curr_frame, kp1,
                                               matches[:50],
                                               None,
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('image', img_with_matches)

        # Return matched features only
        p0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        p1 = np.float32([kp1[m.trainIdx].pt for m in matches])

        return p0, p1
    
    def match_features_flann(self, kp0, kp1, des0, des1, draw_matches=True):
        '''
        This method matches feature descriptors using FLANN-based method.

        Args:
            kp0: Previous keypoints
            kp1: Current keypoints
            des0: Previous descriptors
            des1: Current descriptors
            draw_matches: Whether to display feature matches
        '''
        # Use FLANN-based matcher
        index_params = dict(algorithm=self.flann_index, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des0, des1, k=2)

        # Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < self.threshold * n.distance:
                    good.append(m)
        except ValueError:
            pass

        if draw_matches:
            img_with_matches = cv2.drawMatches(self.prev_frame, kp0,
                                               self.curr_frame, kp1,
                                               good[:100], None,
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('image', img_with_matches)

        # Return features with good matches per ratio test
        p0 = np.float32([kp0[m.queryIdx].pt for m in good])
        p1 = np.float32([kp1[m.trainIdx].pt for m in good])

        return p0, p1
    
    def track_features_optical_flow(self, kp0, draw_matches=True):
        '''
        This method matches feature keypoints using Lucas-Kanade optical flow method.

        Args:
            kp0: Previous keypoints
            draw_matches: Whether to display feature matches
        '''
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Calculate optical flow using Lucas-Kanade method
        kp1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame,
                                                  self.curr_frame,
                                                  kp0, None,
                                                  **lk_params)
        
        # Select good matches
        if kp1 is not None:
            good_old = kp0[status==1]
            good_new = kp1[status==1]

        if draw_matches:
            colors = np.random.randint(0, 255, (100, 3))
            # Draw tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(np.int32)
                c, d = old.ravel().astype(np.int32)
                mask = cv2.line(img=np.zeros_like(self.prev_frame), pt1=(a,b), pt2=(c,d),
                                color=colors[i % 100].tolist(), thickness=2)
                frame = cv2.circle(img=self.curr_frame, center=(a,b), radius=5,
                                color=colors[i % 100].tolist(), thickness=-1)

            # Display modified frame
            img_with_mask = cv2.add(frame, mask)
            cv2.imshow('image', img_with_mask)

        return good_old, good_new
    
    def estimate_ego_motion(self, p0, p1):
        '''
        This method recovers a transformation matrix that rotates and translates
        pose from a previous frame into the current frame using previously matched
        keypoint features and preset camera intrinsic parameter values.

        Args:
            p0: Previous matched features
            p1: Current matched features
        '''
        # Nister 5-point algorithm requires at least 5 points to recover pose
        if len(p0) <= 5 or len(p1) <= 5:
            return np.eye(4)

        # Estimate the essential matrix
        E, _ = cv2.findEssentialMat(p0, p1, self.K,
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=1.0)
        
        # Recover relative pose from essential matrix
        _, R, t, _ = cv2.recoverPose(E, p0, p1, self.K)
        
        # Concatenate rotation and translation matrices into single transformation matrix
        tf = np.eye(4, dtype=np.float32)
        tf[:3, :3] = R
        tf[:3, 3] = t.squeeze()
        return tf
    
    def initialize(self, frame):
        '''
        This method initializes the current frame to be shifted into previous frame
        at next loop iteration. Additionally, it writes the initial pose to log:

                     R     t
               |  1  0  0  0  |
        Pose = |  0  1  0  0  |
               |  0  0  1  0  |

        Rotation is identity matrix. Translation is null vector.

        Args:
            frame: Initial image frame in sequence
        '''
        self.curr_frame = frame
        if self.write_to_log:
            with open(self.output_filename, 'w') as outfile:
                mat = self.homogenous_pose[:3,:4]
                np.savetxt(outfile, mat, fmt='%-7.2f')

    def perform_matching_visual_odometry(self, detect, match):
        '''
        This method performs reconstructs relative pose estimates from
        feature matching-based algorithms. It requires the use of callable
        methods for feature detection and feature matching.

        Args:
            detect: Callable method to detect features (keypoints & descriptors)
            match: Callable method to match features
        '''
        kp0, des0 = detect(self.prev_frame)
        kp1, des1 = detect(self.curr_frame)
        p0, p1 = match(kp0, kp1, des0, des1)
        tf = self.estimate_ego_motion(p0, p1)
        self.homogenous_pose = self.homogenous_pose @ np.linalg.inv(tf)
        self.estimated_poses.append(self.homogenous_pose[:3, :4])
    
    def perform_optical_flow_visual_odometry(self, detect):
        '''
        This method performs reconstructs relative pose estimates from
        optical flow-based algorithms. It requires the use of callable
        methods for feature detection.

        Args:
            detect: Callable method to detect features (keypoints only)
        '''
        kp0 = detect(self.prev_frame)
        p0, p1 = self.track_features_optical_flow(kp0)
        tf = self.estimate_ego_motion(p0, p1)
        self.homogenous_pose = self.homogenous_pose @ np.linalg.inv(tf)
        self.estimated_poses.append(self.homogenous_pose[:3, :4])

    def run(self):
        '''
        This method iterates over loaded input images and performs
        the monocular visual odometry sequence.
        '''
        for i, image in enumerate(self.images):
            # Initial frame won't have a previous frame yet
            if i == 0:
                self.initialize(image)
                continue

            # Shift frame using a sliding window approach
            self.prev_frame = self.curr_frame
            self.curr_frame = image

            # Perform visual odometry depending on which method was selected
            self.perform_matching_visual_odometry(detect=self.detect_method,
                                                  match=self.match_method)

            # Write initial pose to log file
            if self.write_to_log:
                with open(self.output_filename, 'a') as outfile:
                    mat = self.homogenous_pose[:3,:4]
                    np.savetxt(outfile, mat, fmt='%-7.2f')

            # OpenCV window control
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                # Wait until another key is pressed
                cv2.waitKey(-1)
        
        # Convert to numpy array
        self.estimated_poses = np.array(self.estimated_poses)

    def compute_total_error(self):
        '''
        This method tracks the total cumulative error between estimated pose and ground
        truth pose over the entire sequence of images.
        
        NOTE: Only position (x,y,z) is considered as part of this calculation.
        '''
        cumulative_error = 0.
        for i in range(len(self.estimated_poses)):
            error = np.linalg.norm(self.estimated_poses[i][:,3] - self.ground_truth_poses[i][:,3])
            cumulative_error += error
        return cumulative_error

# ==============================================================================
# -- plot_rotation_matrix() ----------------------------------------------------
# ==============================================================================

def plot_rotation_matrix(R):
    '''
    This function plots a rotation matrix relative to an identity basis
    for testing and evaluation purposes.

    Args:
        R: Rotation matrix from pose
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original basis vectors
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')

    # Rotated basis vectors
    rotated_basis = R @ np.eye(3)
    ax.quiver(0, 0, 0, *rotated_basis[:, 0], color='m', label='X\'')
    ax.quiver(0, 0, 0, *rotated_basis[:, 1], color='y', label='Y\'')
    ax.quiver(0, 0, 0, *rotated_basis[:, 2], color='c', label='Z\'')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    # Parse input arguments
    argparser = argparse.ArgumentParser(description='Performs visual odometry estimation on *.avi files.')
    argparser.add_argument(
        '--dataset',
        default='KITTI_sequence_02',
        type=str,
        help='KITTI dataset to test against (default: ./KITTI_sequence_2)')
    argparser.add_argument(
        '--nimages',
        default=500,
        type=int,
        help='Max number of images to load from KITTI dataset (default: 500)')
    argparser.add_argument(
        '--feature',
        default='ORB',
        type=str,
        choices=['FAST', 'SIFT', 'ORB'],
        help='Algorithm to use for detecting feature keypoints (default: ORB)')
    argparser.add_argument(
        '--nfeatures',
        default='3000',
        type=int,
        help='Max number of features allowed to detect (default: 3000)')
    argparser.add_argument(
        '--matcher',
        default='FLANN',
        type=str,
        choices=['BF', 'FLANN'],
        help='Algorithm to use for matching feature keypoints (default: FLANN)')
    argparser.add_argument(
        '--threshold',
        default=0.8,
        type=float,
        help='Threshold to use when performing Lowe\'s ratio test (FLANN-based ONLY) (default: 0.8)')
    argparser.add_argument(
        '--method',
        default='matching',
        type=str,
        choices=['matching', 'optical_flow'],
        help='Matching-based VO or Optical-flow-based VO (default: matching)')
    args = argparser.parse_args()

    # Perform visual odometry sequence
    try:
        VO = VisualOdometry(args)
        start = time.time()
        VO.run()
        end = time.time()
        np.set_printoptions(suppress=True)
        print(f'Sequence complete! Cumulative Error: {VO.compute_total_error().round(3)}')
        print(f'Total execution time: {round(end - start, 2)} seconds')
    finally:
        cv2.destroyAllWindows()

    # Plot results (aerial view only)
    plt.plot(VO.ground_truth_poses[:,:,3][:,0], # Get x position from each transform
             VO.ground_truth_poses[:,:,3][:,2], # Get z position from each transform
             label='Ground truth')
    plt.plot(VO.estimated_poses[:,:,3][:,0], # Get x position from each transform
             VO.estimated_poses[:,:,3][:,2], # Get z position from each transform
             label='Estimated')

    plt.title(f'{args.dataset}: Ground Truth vs Prediction')
    plt.xlabel('x position (m)')
    plt.ylabel('z position (m)')
    plt.legend()
    plt.grid()
    plt.show()
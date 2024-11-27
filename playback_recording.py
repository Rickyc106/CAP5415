#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def display_unmodified(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back to beginning
            continue

        # Display unmodified image
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) == ord('q'):
            break

def display_with_harris_corners(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back to beginning
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect corners with Harris corner detection
        dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        
        # Threshold corner detection
        frame[dst > 0.001*dst.max()] = [0, 0, 255]

        # Display modified image with corners
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) == ord('q'):
            break

def display_with_sift_features(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back to beginning
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create SIFT feature detector
        sift = cv2.SIFT_create()

        # Find and draw keypoint features
        kp = sift.detect(gray, None)
        modified = cv2.drawKeypoints(gray, kp, frame, color=(255,0,0))

        # Display modified image with keypoints
        cv2.imshow('Video', modified)
        if cv2.waitKey(25) == ord('q'):
            break  

def display_with_fast_features(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back to beginning
            continue

        # Create FAST feature detector
        fast = cv2.FastFeatureDetector_create()

        # Find and draw keypoint features
        kp = fast.detect(frame, None)
        modified = cv2.drawKeypoints(frame, kp, None, color=(255,0,0))

        # Display modified image with keypoints
        cv2.imshow('Video', modified)
        if cv2.waitKey(25) == ord('q'):
            break

def display_with_good_features(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back to beginning
            continue

        # Find and draw corner features
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)

        # Draw circles
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 3, (255,0,0), -1)

        # Display modified image with keypoints
        cv2.imshow('Video', img)
        if cv2.waitKey(25) == ord('q'):
            break

def display_with_feature_tracking(cap):
    while cap.isOpened():
        ret, frame = cap.read()

        # Continue to next frame if reading first frame
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
            previous_frame = frame
            continue
        # Loop back to beginning if reached last frame
        elif not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Create FAST feature detector
        fast = cv2.FastFeatureDetector_create()

        # Find and draw keypoint features
        kp1 = fast.detect(previous_frame, None)
        kp2 = fast.detect(frame)
        modified = cv2.drawKeypoints(frame, kp2, None, color=(255,0,0))

        # Display modified image with keypoints
        cv2.imshow('Video', modified)
        if cv2.waitKey(25) == ord('q'):
            break

        # Set current frame as previous for next iteration
        previous_frame = frame

def display_feature_matching_with_brute_force(cap):
    ret, frame1 = cap.read()
    if not ret:
        raise('Failed to read first image')
    
    ret, frame2 = cap.read()
    if not ret:
        raise('Failed to read second image')
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints1, descriptor1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_of_matches = 200
    matched_img = cv2.drawMatches(img1=gray1, keypoints1=keypoints1,
                                  img2=gray2, keypoints2=keypoints2,
                                  matches1to2=matches[:num_of_matches],
                                  outImg=None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(matched_img)
    plt.show()

def display_feature_matching_with_knn_brute_force(cap):
    ret, frame1 = cap.read()
    if not ret:
        raise('Failed to read first image')
    
    ret, frame2 = cap.read()
    if not ret:
        raise('Failed to read second image')
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    num_of_neighbors = 2
    matches = bf.knnMatch(descriptor1, descriptor2, k=num_of_neighbors)
    good_matches = []
    test_ratio = 0.05

    for m, n in matches:
        if m.distance < test_ratio * n.distance:
            good_matches.append([m])

    matched_img = cv2.drawMatchesKnn(img1=gray1, keypoints1=keypoints1,
                                     img2=gray2, keypoints2=keypoints2,
                                     matches1to2=good_matches, outImg=None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(matched_img)
    plt.show()

def display_feature_tracking_with_LK_matching(cap):
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    random_colors = np.random.randint(low=0,
                                      high=255,
                                      size=(100, 3))

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Continue to next frame if reading first frame
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
            prev_gray = gray
            prev_kp = cv2.goodFeaturesToTrack(prev_gray,
                                              maxCorners=100,
                                              qualityLevel=0.3,
                                              minDistance=7)
            mask = np.zeros_like(frame)
            continue
        # Loop back to beginning if reached last frame
        elif not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Calculate optical flow using Lucas-Kanade method
        kp, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_kp, None, **lk_params)

        # Select good matches
        if kp is not None:
            good_new = kp[st==1]
            good_old = prev_kp[st==1]

        # Draw tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(np.int32)
            c, d = old.ravel().astype(np.int32)
            mask = cv2.line(img=mask, pt1=(a,b), pt2=(c,d),
                            color=random_colors[i].tolist(), thickness=2)
            frame = cv2.circle(img=frame, center=(a,b), radius=5,
                               color=random_colors[i].tolist(), thickness=-1)

        # Display modified frame
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_kp = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(100) == ord('q'):
            break

def perform_visual_odometry(args):
    try:
        cap = cv2.VideoCapture(f'data/output-{args.index}.avi')
        if not cap.isOpened():
            raise('Error reading video file')
        display_feature_tracking_with_LK_matching(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plays back recorded *.avi files.')
    argparser.add_argument(
        '--index',
        default=0,
        type=int,
        help='Index number (default: 0)')
    args = argparser.parse_args()
    perform_visual_odometry(args)
#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

class FeatureMatcher(object):
    def __init__(self):
        self.detector = cv2.ORB_create()

    def extract_features(self, img):
        '''
        img: Single image
        '''
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints / descriptors
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features_brute_force(self, d0, d1):
        '''
        d0: Previous descriptors from ORB detection
        d1: Current descriptors
        '''
        # Use brute-force matching with hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d0, d1)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches
    
    def match_features_flann(self, d0, d1):
        '''
        d0: Previous descriptors from ORB detection
        d1: Current descriptors
        '''
        # Use FLANN-based matching
        FLANN_INDEX_KDTREE = 1
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        matches = flann.knnMatch(d0, d1, k=2)

        # Build a mask to filter out bad matches
        mask = [[0,0] for i in range(len(matches))]

        # Perform ratio test per Lowe's paper (???)
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.4 * n.distance:
                mask[i] = [1,0]

        # Use mask to specify which line matches to draw
        drawParams = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = mask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)

        return matches, drawParams

    def match(self, img1, img2):
        # Extract ORB features / descriptors from each image
        kp0, d0 = self.extract_features(img1)
        kp1, d1 = self.extract_features(img2)

        # Find best matches for descriptors between images
        matches = self.match_features_brute_force(d0, d1)
        # matches, drawParams = self.match_features_flann(d0, d1)

        return kp0, kp1, matches

    def display_matches(self, img1, img2, kp0, kp1, matches):
        # Display modified frame
        img_with_matches = cv2.drawMatches(img1, kp0,
                                           img2, kp1,
                                           matches[:40],
                                           None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # img_with_matches = cv2.drawMatchesKnn(img1, kp0,
        #                                       img2, kp1,
        #                                       matches, None,
        #                                       **drawParams)
        cv2.imshow('matches', img_with_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse input arguments
    argparser = argparse.ArgumentParser(description='Performs feature matching on two sample images.')
    args = argparser.parse_args()

    # Read sample images
    img1 = cv2.imread('./test-images/1.png')
    img2 = cv2.imread('./test-images/2.png')

    # Perform feature matching algorithm
    FM = FeatureMatcher()
    kp0, kp1, matches = FM.match(img1, img2)

    # Display results
    FM.display_matches(img1, img2, kp0, kp1, matches)
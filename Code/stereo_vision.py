#!/usr/env/bin python3

"""
ENPM673 Spring 2022: Perception for Autonomous Robots

Project 3: Stereo Vision

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing modules
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import time
from typing import List, Tuple

from utils import *


class StereoVision:

    def __init__(self, data_dir: str) -> None:
        self.data_dir = get_data_files(data_dir)
        self.calib_params = parse_params(self.data_dir[0])
        self.img_set = create_image_set(self.data_dir[1:])

    def __get_matches(self, img0: np.array, img1: np.array, visualize: bool=False) -> Tuple[np.array, np.array]:
        # Reference - https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

        print("Estimating feature pairs...")

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img0, None)
        kp2, des2 = sift.detectAndCompute(img1, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matchesMask = [[0,0] for i in range(len(matches))]

        self.x0 = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1,2) # (width, height)
        self.x1 = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1,2) # (width, height)

        print("Found {} feature pairs.".format(len(matches)))

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.5*n.distance:
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv2.DrawMatchesFlags_DEFAULT)

        img_matches = cv2.drawMatchesKnn(img0, kp1, img1, kp2, matches, None, **draw_params)

        if(visualize):
            # cv2.imshow("Inputs", np.hstack((img0, img1)))
            cv2.imshow("Matches", img_matches)
            cv2.waitKey()

        return self.x0, self.x1

    def __estimate_F_mat(self, x0: np.array, x1: np.array) -> np.array:
        
        def construct_A(x0: np.array, x1: np.array) -> np.array:
            A = np.empty([0,9])

            for i, j in zip(x0, x1):
                A_ij = np.array([i[0]*j[0], i[0]*j[1], i[0], i[1]*j[0], i[1]*j[1], i[1], j[0], j[1], 1]).reshape(1,9)
                A = np.append(A, A_ij, axis=0)

            return A

        A = construct_A(x0, x1)

    def calibrate(self):
        x0, x1 = self.__get_matches(self.img_set[0], self.img_set[1], False)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataDir', type=str, default="../Data/curule/", help='Path to the input data directory')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')

    Args = Parser.parse_args()
    data_dir = Args.DataDir
    visualize = Args.Visualize

    SV = StereoVision(data_dir)
    SV.calibrate()

if __name__ == '__main__':
    main()
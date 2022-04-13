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
from tqdm import tqdm

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

    def __estimate_F_mat(self, x0: np.array, x1: np.array, T0: np.array, T1: np.array) -> np.array:
        # Reference - https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html

        def construct_A(x0: np.array, x1: np.array) -> np.array:
            A = np.empty([0,9])

            for i, j in zip(x0, x1):
                A_ij = np.array([i[0]*j[0], i[0]*j[1], i[0], i[1]*j[0], i[1]*j[1], i[1], j[0], j[1], 1]).reshape(1,9)
                A = np.append(A, A_ij, axis=0)

            return A

        A = construct_A(x0, x1)
        U, S, V_t = np.linalg.svd(A)
        F = V_t[-1].reshape(3,3)

        U_f, S_f, V_t_f = np.linalg.svd(F)
        S_f_ = np.diag(S_f)
        S_f_[2,2] = 0
        F_ = np.dot(U_f, np.dot(S_f_, V_t_f))

        F = np.dot(T1.transpose(), np.dot(F_, T0))
        F = F/F[-1,-1]

        return F

    def __RANSAC_F_mat(self, x0: np.array, x1: np.array, epsilon: float, iterations: int) -> Tuple[np.array, np.array]:

        max_inliers = 0
        best_inliers = None
        best_F = None
        features = np.arange(len(x0)).tolist()

        def normalize_features(x: np.array) -> np.array:
            # Reference - https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html

            x_u_, x_v_ = np.mean(x, axis=0)
            x_ = x - [x_u_, x_v_]
            s = np.sqrt(np.mean(x_[:,0]**2 + x_[:,1]**2))
            T_s = np.diag([s, s, 1])
            T_t = np.array([[1, 0, -x_u_],
                           [0, 1, -x_v_],
                           [0, 0, 1]])
            T = np.dot(T_s, T_t)
            x_ = np.column_stack((x_, np.ones(len(x_)))).transpose()
            x_hat = np.dot(T, x_).transpose()

            return x_hat, T

        print("Performing RANSAC to estimate best F...")
        for itr in tqdm(range(iterations)):
            inliers = list()
            feature_pairs = np.random.choice(features, 8, replace=False)
            x0_, T0 = normalize_features(x0[feature_pairs])
            x1_, T1 = normalize_features(x1[feature_pairs])
            F = self.__estimate_F_mat(x0_, x1_, T0, T1)

            for feature in range(len(features)):
                x0_ = np.array([x0[feature][0], x0[feature][1], 1]).reshape(3,1)
                x1_ = np.array([x1[feature][0], x1[feature][1], 1]).reshape(3,1)
                epi_const = np.dot(np.dot(x1_.transpose(), F), x0_)
                if(np.abs(epi_const) < epsilon):
                    inliers.append([x0_, x1_])

            if(len(inliers) >= max_inliers):
                max_inliers = len(inliers)
                best_inliers = inliers
                best_F = F

        best_inliers = np.array(best_inliers)
        # print(best_inliers, best_inliers.shape)
        print(F)

        return F

    def calibrate(self):
        x0, x1 = self.__get_matches(self.img_set[0], self.img_set[1], False)
        F = self.__RANSAC_F_mat(x0, x1, 0.01, 2000)


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
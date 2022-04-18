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
import seaborn as sns
sns.set_theme()

from utils import *

np.set_printoptions(precision=3)

class StereoVision:

    def __init__(self, data_dir: str, save: bool=False, visualize: bool=False) -> None:
        self.data_dir = data_dir
        self.data_files = get_data_files(data_dir)
        self.dataset = self.data_dir.split('/')[-2]
        print("Dataset: {}".format(self.dataset))
        self.save = save
        self.visualize = visualize
        self.save_path = os.path.join('../Results', self.dataset)
        if(self.save):
            if(not os.path.exists(self.save_path)):
                os.makedirs(self.save_path, exist_ok=True)

        self.calib_params = parse_params(self.data_files[0])
        self.K = [self.calib_params['K0'], self.calib_params['K1']]
        self.img_set = create_image_set(self.data_files[1:])

    def __get_matches(self, img0: np.array, img1: np.array, visualize: bool=False) -> Tuple[np.array, np.array]:
        # Reference - https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

        print("\nEstimating feature pairs...")

        sift = cv2.SIFT_create()

        img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(img0_gray, None)
        kp2, des2 = sift.detectAndCompute(img1_gray, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matchesMask = [[0,0] for i in range(len(matches))]

        x0 = np.empty([0,2])
        x1 = np.empty([0,2])

        for i,(m,n) in enumerate(matches):
            if m.distance < (0.8 * n.distance):
                x0 = np.append(x0, np.array([kp1[m.queryIdx].pt]), axis=0)
                x1 = np.append(x1, np.array([kp2[m.trainIdx].pt]), axis=0)
                matchesMask[i]=[1,0]

        print("Found {} feature pairs.".format(len(x0)))

        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv2.DrawMatchesFlags_DEFAULT)

        img_matches = cv2.drawMatchesKnn(img0, kp1, img1, kp2, matches, None, **draw_params)

        if(self.visualize):
            # cv2.imshow("Inputs", np.hstack((img0, img1)))
            cv2.imshow("Matches", img_matches)
            cv2.waitKey()

        if(self.save):
            cv2.imwrite(os.path.join(self.save_path, self.dataset + '_matches.png'), img_matches)

        self.x0, self.x1 = x0, x1
        return self.x0, self.x1

    def __estimate_F_mat(self, x0: np.array, x1: np.array) -> np.array:
        # Reference - https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html

        def construct_A(x0: np.array, x1: np.array) -> np.array:
            A = np.empty([0,9])

            for i, j in zip(x1, x0):
                A_ij = np.array([i[0]*j[0], i[0]*j[1], i[0], i[1]*j[0], i[1]*j[1], i[1], j[0], j[1], 1]).reshape(1,9)
                A = np.append(A, A_ij, axis=0)

            return A

        A = construct_A(x0, x1)
        U, S, V_t = np.linalg.svd(A, full_matrices=True)
        F = V_t[-1].reshape(3,3)

        U_f, S_f, V_t_f = np.linalg.svd(F)
        S_f_ = np.diag(S_f)
        S_f_[2,2] = 0
        F = np.dot(U_f, np.dot(S_f_, V_t_f)) # F_norm

        return F

    def __RANSAC_F_mat(self, x0: np.array, x1: np.array, epsilon: float, iterations: int, visualize: bool=False) -> Tuple[np.array, np.array]:

        print("\nPerforming RANSAC to estimate best F...")

        max_inliers = 0
        best_inliers = None
        best_F = None
        features = np.arange(len(x0)).tolist()

        def normalize_features(x: np.array) -> np.array:
            # Reference - https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html

            x_u_, x_v_ = np.mean(x, axis=0)
            x_ = x - [x_u_, x_v_]
            s = (2/np.mean(x_[:,0]**2 + x_[:,1]**2))**0.5
            T_s = np.diag([s, s, 1])
            T_t = np.array([[1, 0, -x_u_],
                           [0, 1, -x_v_],
                           [0, 0, 1]])
            T = np.dot(T_s, T_t)
            x_ = np.column_stack((x, np.ones(len(x))))
            x_hat = np.dot(T, x_.transpose()).transpose()

            return x_hat, T

        x0_norm, T0 = normalize_features(x0)
        x1_norm, T1 = normalize_features(x1)

        for itr in tqdm(range(iterations)):
            inliers = list()
            feature_pairs = np.random.choice(features, 8, replace=False)

            x0_ = x0_norm[feature_pairs]
            x1_ = x1_norm[feature_pairs]
            F_norm = self.__estimate_F_mat(x0_, x1_)
            # print("F: \n", F)

            x0_ = np.vstack((x0_norm[:,0], x0_norm[:,1], np.ones(len(x0_norm))))
            x1_ = np.vstack((x1_norm[:,0], x1_norm[:,1], np.ones(len(x1_norm))))
            epi_const = np.dot(np.dot(x1_.transpose(), F_norm), x0_)
            # print(epi_const)
            epi_const = np.abs(np.diagonal(epi_const))
            # print(epi_const, epi_const.shape)

            inliers_idx = np.where(epi_const <= epsilon)
            x0_inliers = x0[inliers_idx[0]]
            x1_inliers = x1[inliers_idx[0]]
            inliers = [x0_inliers, x1_inliers]

            # input('q')
            if(len(x0_inliers) >= max_inliers):
                max_inliers = len(x0_inliers)
                best_inliers = inliers
                best_F = F_norm

        best_inliers = np.array(best_inliers)
        print("Found {} inliers from RANSAC".format(len(best_inliers[0])))

        temp = np.hstack((np.copy(self.img_set[0]), np.copy(self.img_set[1])))
        for x0, x1 in zip(best_inliers[0], best_inliers[1]):
            cv2.circle(temp,(int(x0[0]), int(x0[1])),2,(0,0,255), 2)
            cv2.circle(temp,(int(x1[0])+self.img_set[0].shape[1], int(x1[1])),2,(0,0,255), 2)
            cv2.line(temp, (int(x0[0]), int(x0[1])), (int(x1[0])+self.img_set[0].shape[1], int(x1[1])), (0,255,0), 1)

        if(self.visualize):
            cv2.imshow("Inliers", temp)
            cv2.waitKey()

        if(self.save):
            cv2.imwrite(os.path.join(self.save_path, self.dataset + '_inliers.png'), temp)

        x0_norm, T0 = normalize_features(best_inliers[0])
        x1_norm, T1 = normalize_features(best_inliers[1])
        F_norm = self.__estimate_F_mat(x0_norm, x1_norm)
        F = np.dot(T1.transpose(), np.dot(F_norm, T0))
        # F = F/F[-1,-1]

        return F, best_inliers

    def __estimate_E_mat(self, F: np.array, K: list) -> np.array:
        E = np.dot(K[1].transpose(), np.dot(F, K[0]))

        U, S, V_t = np.linalg.svd(E)
        S = np.diag([1, 1, 0])
        E = np.dot(U, np.dot(S, V_t))
        return E

    def __estimate_cam_pose(self, E: np.array, K: list, x0: np.array, x1: np.array) -> np.array:

        print("\nEstimating camera pose...")

        def projection_mat(K: np.array, R: np.array, C: np.array) -> np.array:
            I = np.eye(3)
            P = np.empty([0, 3, 4])
            for i in range(len(R)):
                P_i = np.dot(K, np.dot(R[i], np.hstack((I, -C[i].reshape(3,1))))).reshape(1,3,4)
                P = np.append(P, P_i, axis=0)

            return P

        def compute_triangulated_pts(x0: np.array, x1: np.array, K: list, P: np.array):

            R = np.identity(3)
            C = np.zeros((3,1))
            I = np.identity(3)
            P0 = np.dot(K[0], np.dot(R, np.hstack((I, -C))))

            triangulated_pts_set = list()
            for P_i in P:
                triangulated_pts = np.empty([0,4])
                for p1, p2 in zip(x0, x1):
                    x = np.dot(p1[0], P0[2].reshape(1,-1)) - P0[0]
                    y = np.dot(p1[1], P0[2].reshape(1,-1)) - P0[1]
                    x_ = np.dot(p2[0], P_i[2].reshape(1,-1)) - P_i[0]
                    y_ = np.dot(p2[1], P_i[2].reshape(1,-1)) - P_i[1]

                    A = np.vstack((x, y, x_, y_))
                    U, D, V_t = np.linalg.svd(A)
                    pt = (V_t[-1]/V_t[-1,-1]).reshape(1,-1)
                    triangulated_pts = np.append(triangulated_pts, pt, axis=0)
                triangulated_pts_set.append(triangulated_pts)

            triangulated_pts_set = np.array(triangulated_pts_set)
            return triangulated_pts_set

        def disambiguate_cam_poses(R: np.array, C: np.array, triangulated_pts_set: np.array):

            max_pts = list()
            for i in range(len(R)):
                pts = 0
                for pt in triangulated_pts_set[i]:
                    if(np.dot(R[i][2,:], (pt.reshape(4,1)[:3]  - C[i]))[0] > 0):
                        pts += 1
                max_pts.append(pts)

            return np.argmax(max_pts)

        U, D, V_t = np.linalg.svd(E)
        W = np.float32(np.array([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]]))

        R = np.empty([0,3,3])
        R = np.append(R, np.dot(U, np.dot(W, V_t)).reshape(1,3,3), axis=0)
        R = np.append(R, np.dot(U, np.dot(W, V_t)).reshape(1,3,3), axis=0)
        R = np.append(R, np.dot(U, np.dot(W.transpose(), V_t)).reshape(1,3,3), axis=0)
        R = np.append(R, np.dot(U, np.dot(W.transpose(), V_t)).reshape(1,3,3), axis=0)

        C = np.empty([0,3,1])
        C = np.append(C, U[:,2].reshape(1,3,1), axis=0)
        C = np.append(C, -U[:,2].reshape(1,3,1), axis=0)
        C = np.append(C, U[:,2].reshape(1,3,1), axis=0)
        C = np.append(C, -U[:,2].reshape(1,3,1), axis=0)

        for i in range(len(R)):
            if(np.linalg.det(R[i]) < 0):
                # print("Correcting R, C sign...")
                R[i] = -R[i]
                C[i] = -C[i]

        P = projection_mat(K[1], R, C)
        triangulated_pts_set = compute_triangulated_pts(x0, x1, K, P)
        idx = disambiguate_cam_poses(R, C, triangulated_pts_set)

        return R[idx], C[idx]

    def draw_epi_lines(self, img: np.array, lines: np.array, pts: np.array) -> np.array:
        r, c = img.shape[:2]

        epi_lines = np.random.choice(np.arange(len(lines)), 100, replace=False)
        for r, pt in zip(lines[epi_lines], pts[epi_lines]):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1]])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img = cv2.line(img, (x0,y0), (x1,y1), color,2)
            img = cv2.circle(img,tuple(pt[0]),5,color,-1)

        return img

    def calibrate(self):
        x0, x1 = self.__get_matches(self.img_set[0], self.img_set[1], False)
        self.F, self.inliers = self.__RANSAC_F_mat(x0, x1, 0.002, 2000, False)
        print("F:\n", self.F)
        self.E = self.__estimate_E_mat(self.F, self.K)
        print("E:\n", self.E)
        self.R, self.t = self.__estimate_cam_pose(self.E, self.K, self.inliers[0], self.inliers[1])
        print("R:\n", self.R)
        print("t:\n", self.t)

    def rectify(self, visualize: bool=False) -> Tuple[np.array, np.array]:
        # Reference - https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

        print("\nRectifying the stereo pair...")
        h, w = self.img_set[0].shape[:2]

        ret, H1, H2 = cv2.stereoRectifyUncalibrated(self.inliers[0], self.inliers[1], self.F, (w, h))

        print("H_l:\n", H1)
        print("H_r:\n", H2)

        img0_rect = cv2.warpPerspective(self.img_set[0], H1, (w, h))
        img1_rect = cv2.warpPerspective(self.img_set[1], H2, (w, h))

        H2_T_inv =  np.linalg.inv(H2.transpose())
        H1_inv = np.linalg.inv(H1)
        F_rect = np.dot(H2_T_inv, np.dot(self.F, H1_inv))

        # Unrectified
        epi_unrect_lines0 = cv2.computeCorrespondEpilines(self.inliers[0].reshape(-1,1,2), 2, F_rect).reshape(-1,3)
        epi_unrect_lines1 = cv2.computeCorrespondEpilines(self.inliers[1].reshape(-1,1,2), 1, F_rect).reshape(-1,3)

        img0_unrect_epi_lines = self.draw_epi_lines(np.copy(self.img_set[0]), epi_unrect_lines0, self.inliers[0].reshape(-1,1,2).astype(np.int32))
        img1_unrect_epi_lines = self.draw_epi_lines(np.copy(self.img_set[1]), epi_unrect_lines1, self.inliers[1].reshape(-1,1,2).astype(np.int32))

        x0_trans = cv2.perspectiveTransform(np.float32(self.inliers[0]).reshape(-1,1,2), H1)
        x1_trans = cv2.perspectiveTransform(np.float32(self.inliers[1]).reshape(-1,1,2), H2)

        epi_lines0 = cv2.computeCorrespondEpilines(x1_trans.reshape(-1,1,2), 2, F_rect).reshape(-1,3)
        epi_lines1 = cv2.computeCorrespondEpilines(x0_trans.reshape(-1,1,2), 1, F_rect).reshape(-1,3)

        img0_epi_lines = self.draw_epi_lines(np.copy(img0_rect), epi_lines0, x0_trans.astype(np.int32))
        img1_epi_lines = self.draw_epi_lines(np.copy(img1_rect), epi_lines1, x1_trans.astype(np.int32))


        if(self.visualize):
            cv2.imshow("IMGs Unrectified", np.hstack((img0_unrect_epi_lines, img1_unrect_epi_lines)))
            cv2.imshow("IMGs Rectified", np.hstack((img0_epi_lines, img1_epi_lines)))
            cv2.waitKey()

        if(self.save):
            cv2.imwrite(os.path.join(self.save_path, self.dataset + '_rectified.png'), np.hstack((img0_rect, img1_rect)))
            cv2.imwrite(os.path.join(self.save_path, self.dataset + '_unrectified_epi_lines.png'), np.hstack((img0_unrect_epi_lines, img1_unrect_epi_lines)))
            cv2.imwrite(os.path.join(self.save_path, self.dataset + '_epi_lines.png'), np.hstack((img0_epi_lines, img1_epi_lines)))

        self.img_rect_set = [img0_rect, img1_rect]
        return img0_rect, img1_rect

    def compute_disparity(self, block_size: int, search_window: int=0, SSD: bool=False, visualize: bool=False) -> np.array:
        # Reference - https://pramod-atre.medium.com/disparity-map-computation-in-python-and-c-c8113c63d701
        #           - http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/

        print("\nComputing disparity...")

        img_l = cv2.cvtColor(self.img_rect_set[0], cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(self.img_rect_set[1], cv2.COLOR_BGR2GRAY)
        
        if(visualize):
            cv2.imshow("Stereo Pair", np.hstack((img_l, img_r)))
            cv2.waitKey()

        h, w = img_l.shape
        disparity_map = np.zeros((h, w))

        for r in tqdm(range(0, h-block_size)):
            left_row = list()
            right_row = list()
            for c in range(0, w-block_size):
                left_block = img_l[r:r+block_size,c:c+block_size].flatten()
                left_row.append(left_block)

                right_block = img_r[r:r+block_size,c:c+block_size].flatten()
                right_row.append(right_block)

            left_row = np.array(left_row)
            right_row = np.array(right_row)

            for i in range(len(left_row)):
                start = max(0, i-search_window)
                end = min(len(right_row), i+search_window)
                if(SSD):
                    similarity_scores = np.sum((right_row[start:end] - left_row[i])**2, axis=1)
                else:
                    similarity_scores = np.sum(np.abs(right_row[start:end] - left_row[i]), axis=1)
                idx = np.argmin(similarity_scores)
                disparity = np.abs(i - (start+idx))
                disparity_map[r,i] = disparity

        # disparity_map = normalize(disparity_map, 0, 255)
        disparity_map = np.interp(disparity_map, (disparity_map.min(), disparity_map.max()), (0, 255)).astype(np.uint8)

        plt.figure(figsize=(6.4,3.6), dpi=100)
        plt.axis('off')
        sns.heatmap(disparity_map, cmap='gray')
        # plt.imshow(disparity_map, cmap='gray', interpolation='nearest')

        if(self.visualize):
            plt.show()

        if(self.save):
            plt.savefig(os.path.join(self.save_path, self.dataset + '_disparity_gray.png'), bbox_inches='tight')
            plt.clf()
            plt.figure(figsize=(6.4,3.6), dpi=100)
            plt.axis('off')
            sns.heatmap(disparity_map)
            plt.savefig(os.path.join(self.save_path, self.dataset + '_disparity_heatmap.png'), bbox_inches='tight')
            # with open (os.path.join(self.save_path, self.dataset + '_disparity.npy'), 'wb') as f:
            #     np.save(f, disparity_map)

        self.disparity_map = disparity_map
        return disparity_map

    def compute_depth(self, visualize: bool=False):

        print("\nComputing depth map...")

        baseline = self.calib_params['baseline']
        f = self.K[0][0,0]

        depth_map = (baseline * f) / (self.disparity_map + 1e-10)

        depth_map[depth_map > np.median(depth_map)] = np.median(depth_map)
        # depth_map = normalize(depth_map, 0, 255)
        depth_map = np.interp(depth_map, (depth_map.min(), depth_map.max()), (0, 255)).astype(np.uint8)

        plt.figure(figsize=(6.4,3.6), dpi=100)
        plt.axis('off')
        sns.heatmap(depth_map)
        # plt.imshow(depth_map, cmap='hot', interpolation='bilinear')

        if(self.visualize):
            plt.show()

        if(self.save):
            plt.savefig(os.path.join(self.save_path, self.dataset + '_depth_heatmap.png'), bbox_inches='tight')
            plt.clf()
            plt.figure(figsize=(6.4,3.6), dpi=100)
            plt.axis('off')
            sns.heatmap(depth_map, cmap='gray')
            plt.savefig(os.path.join(self.save_path, self.dataset + '_depth_gray.png'), bbox_inches='tight')

        self.depth_map = depth_map
        return depth_map

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataDir', type=str, default="../Data/curule/", help='Path to the input data directory')
    Parser.add_argument('--Save', action='store_true', help='Toggle saving results')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')

    Args = Parser.parse_args()
    data_dir = Args.DataDir
    save = Args.Save
    visualize = Args.Visualize

    SV = StereoVision(data_dir, save, visualize)
    SV.calibrate()
    SV.rectify()
    SV.compute_disparity(21, 75, True)
    SV.compute_depth()

if __name__ == '__main__':
    main()
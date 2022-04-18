#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Project 3: Stereo Vision

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

import sys
import cv2
import os
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

def normalize(img, _min, _max) -> np.array:
    return np.uint8(cv2.normalize(img, None, _min, _max, cv2.NORM_MINMAX))

def convert_three_channel(img) -> np.array:
    return np.dstack((img, img, img))

def get_data_files(data_dir) -> list:
    return [os.path.join(data_dir,  f) for f in sorted(os.listdir(data_dir))]

def create_image_set(img_set_paths: list) -> np.array:
    img_set = list()
    for img_path in img_set_paths:
        img_set.append(cv2.imread(img_path))

    img_set = np.array(img_set)
    return img_set

def parse_params(calib_path: str) -> dict:
    df = pd.read_csv(calib_path, sep='=', header=None)
    
    params = dict()

    K0 = str(df[1][0]).replace('[', ' ').replace(']', ' ').replace(';', ' ').split()
    K0 = np.array([float(i) for i in K0]).reshape(3,3)
    params['K0'] = K0
    
    K1 = str(df[1][1]).replace('[', ' ').replace(']', ' ').replace(';', ' ').split()
    K1 = np.array([float(i) for i in K1]).reshape(3,3)
    params['K1'] = K1

    for param in range(2, len(df[0])):
        params[df[0][param]] = float(df[1][param])

    return params
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

sys.dont_write_bytecode = True

def normalize(img, _min, _max):
    return np.uint8(cv2.normalize(img, None, _min, _max, cv2.NORM_MINMAX))

def convert_three_channel(img):
    return np.dstack((img, img, img))

def read_image_set(data_dir):
    return [os.path.join(data_dir,  f) for f in sorted(os.listdir(data_dir))]
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

from utils import *


class StereoVision:

	def __init__(self, data_dir: str) -> None:
		self.data_dir = get_data_files(data_dir)
		self.calib_params = parse_params(self.data_dir[0])

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DataDir', type=str, default="../Data/curule/", help='Path to the input data directory')
	Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')

	Args = Parser.parse_args()
	data_dir = Args.DataDir
	visualize = Args.Visualize

	SV = StereoVision(data_dir)

if __name__ == '__main__':
	main()
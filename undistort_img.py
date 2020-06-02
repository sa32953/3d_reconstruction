#!/usr/bin/env python 3.6

########################################
#        Undistorting Images           #
########################################

import cv2 as cv
import numpy as np 
#import matplotlib.pyplot as plt 

#Load Camera Parameters
ret = np.load("/home/sa0102/computer_vision/project/source code/calibration/ret.npy")
K = np.load("/home/sa0102/computer_vision/project/source code/calibration/K.npy")
dist = np.load("/home/sa0102/computer_vision/project/source code/calibration/dist.npy")
rvecs = np.load("/home/sa0102/computer_vision/project/source code/calibration/rvecs.npy")
tvecs = np.load("/home/sa0102/computer_vision/project/source code/calibration/tvecs.npy")

#Read Image
imagepath = '/home/sa0102/computer_vision/project/images/calib_t2/samsung/ct2.jpg'
d_img = cv.imread(imagepath)

#Find Optimal Camera Matrix
h,  w = d_img.shape[:2]
new_cameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

#Undistort Image
und_img = cv.undistort(d_img, K, dist, None, new_cameramtx)

# Show Img
und_img = cv.resize(und_img, None, fx=0.25, fy=0.25, interpolation= cv.INTER_CUBIC)
cv.imshow('u_img', und_img)
cv.waitKey(0)
#!/usr/bin/env python 3.6


import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

# Input 2 images
d_img1 = cv.imread('/home/sa0102/computer_vision/project/images/a.jpg') # queryImage
d_img2 = cv.imread('/home/sa0102/computer_vision/project/images/b.jpg') # trainImage

########################################
#        Undistorting Images           #
########################################

#Load Camera Parameters
ret = np.load("/home/sa0102/computer_vision/project/source code/calibration/ret.npy")
K = np.load("/home/sa0102/computer_vision/project/source code/calibration/K.npy")
dist = np.load("/home/sa0102/computer_vision/project/source code/calibration/dist.npy")
rvecs = np.load("/home/sa0102/computer_vision/project/source code/calibration/rvecs.npy")
tvecs = np.load("/home/sa0102/computer_vision/project/source code/calibration/tvecs.npy")

h,  w = d_img1.shape[:2]
new_cameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

img1 = cv.undistort(d_img1, K, dist, None, new_cameramtx)
img2 = cv.undistort(d_img2, K, dist, None, new_cameramtx)

########################################
#    Feature Description and Matching  #
########################################

# Initiate ORB detector
orb = cv.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# Create BFMatcher object
bf = cv.BFMatcher()

# Match descriptors, cross check in above command already applies ratio test
matches = bf.knnMatch(des1,des2, k=2)

# Ratio Test as per Lowe's paper
good = []
pts1 = []
pts2 = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Draw good matches only.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
# Take care that Plt shows in BGR

# Implemented Ratio test in Lowe's paper

########################################
#           Fundamental Matrix         #
########################################

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Fundamental matrix
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Next step is to calculate normal vector of 2 projecting points and check if F transform 1 pt to other?

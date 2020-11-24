#!/usr/bin/env python 3.6

########################################
#        Created by: Sahil A.          #
#    HOCHSCHULE RAVENSBURG WEINGARTEN  #
########################################

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

########################################
#             Read Images              #
########################################

# Input 2 images
d_img1 = cv.imread('/home/sa0102/computer_vision/project/images/d.jpg', cv.CV_8UC3) # queryImage
d_img2 = cv.imread('/home/sa0102/computer_vision/project/images/c.jpg', cv.CV_8UC3 ) # trainImage

#d_img1 = cv.imread('/home/sa0102/computer_vision/project/dataset/tsukuba/scene1.row3.col3.ppm', cv.CV_8UC3) # queryImage
#d_img2 = cv.imread('/home/sa0102/computer_vision/project/dataset/tsukuba/scene1.row3.col4.ppm', cv.CV_8UC3 ) # trainImage

if len(d_img1.shape) == 2:
    d_img1 = cv.cvtColor(d_img1, cv.COLOR_GRAY2BGR)
    d_img2 = cv.cvtColor(d_img2, cv.COLOR_GRAY2BGR)

########################################
#        Undistorting Images           #
########################################

#Load Camera Parameters
ret = np.load("/home/sa0102/computer_vision/project/source code/calibration/ret.npy")
K = np.load("/home/sa0102/computer_vision/project/source code/calibration/K.npy")
dist = np.load("/home/sa0102/computer_vision/project/source code/calibration/dist.npy")
rvecs = np.load("/home/sa0102/computer_vision/project/source code/calibration/rvecs.npy")
tvecs = np.load("/home/sa0102/computer_vision/project/source code/calibration/tvecs.npy")

target_width = 600
use_pyr_down=True
if use_pyr_down and d_img1.shape[1] > target_width:
    while d_img1.shape[1] > 2*target_width:
            d_img1 = cv.pyrDown(d_img1)
            d_img2 = cv.pyrDown(d_img2)

h,  w = d_img1.shape[:2]
new_cameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

# undistort the images
#img1 = cv.undistort(d_img1, K, dist, None, new_cameramtx)
#img2 = cv.undistort(d_img2, K, dist, None, new_cameramtx)
img1 = d_img1
img2 = d_img2

########################################
#    Feature Description and Matching  #
########################################

# Initiate ORB detector
orb = cv.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None) #query points
kp2, des2 = orb.detectAndCompute(img2,None) #train points

# Create BFMatcher object
bf = cv.BFMatcher()

# Match descriptors, cross check in above command already applies ratio test
matches = bf.knnMatch(des1,des2,k =2)

# Ratio Test as per Lowe's paper
good = []
pts1 = []
pts2 = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt) #train pts
        pts1.append(kp1[m.queryIdx].pt) # query pts

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
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_8POINT)

# use 8 pt algorith to constraint rank of matrix as 2
# F prjects points from query image to train image ie from pts1 to pt2

########################################
#            Essential Matrix          #
########################################

# Essential Matrix contains the information about translation and rotation...
# ...which describe the location of the second camera relative to the first in global coordinates
E = (K.T).dot(F).dot(K)

# Decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
U, S, Vt = np.linalg.svd(E)
W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

K_inv = np.linalg.inv(K) # Inverse of Camera matrix

first_inliers = []
second_inliers = []
for i in range(len(mask)):
    if mask[i]:
        # normalize and homogenize the image coordinates
        first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
        second_inliers.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))


def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

########################################
#          SVD Essential Matrix        #
########################################

R = U.dot(W).dot(Vt)
T = U[:, 2]

# There are 4 possibilities and we chose 1 of it
if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

    # Second choice: R = U * W * Vt, T = -u_3
    T = - U[:, 2]
    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]

        if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):

            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
Rt2 = np.hstack((R, T.reshape(3, 1)))

R = Rt2[:, :3]
T = Rt2[:, 3]


#perform the rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K, dist, K, dist, d_img1.shape[:2], R, T, cv.CALIB_ZERO_DISPARITY, alpha=-1)

mapx1, mapy1 = cv.initUndistortRectifyMap(K, dist, R, new_cameramtx , d_img1.shape[:2], cv.CV_32F)
mapx2, mapy2 = cv.initUndistortRectifyMap(K, dist, R, new_cameramtx , d_img2.shape[:2], cv.CV_32F)

img_rect1 = cv.remap(d_img1, mapx1, mapy1, cv.INTER_CUBIC)
img_rect1_gray = cv.cvtColor(img_rect1, cv.COLOR_BGR2GRAY)
#cv.imshow('r1',img_rect1)
#cv.waitKey(0)
img_rect2 = cv.remap(d_img2, mapx2, mapy2, cv.INTER_CUBIC)
img_rect2_gray = cv.cvtColor(img_rect2, cv.COLOR_BGR2GRAY)
img_rect2 = np.uint8(img_rect2)
#cv.imshow('r2',img_rect2)
#cv.waitKey(0)


# draw the images side by side
total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1], 3)
img = np.zeros(total_size, dtype=np.uint8)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# draw horizontal lines every 25 px accross the side by side image
for i in range(20, img.shape[0], 25):
    cv.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

#cv.imshow('rectified', img)
#cv.waitKey(0)

stereo = cv.StereoBM_create(numDisparities=32, blockSize=11)
disparity = stereo.compute(img_rect1_gray,img_rect2_gray)
plt.imshow(disparity,'gray')
plt.show()
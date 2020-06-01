#!/usr/bin/env python 3.6

########################################
#    Feature Description and Matching  #
########################################

import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

# Input 2 images
img1 = cv.imread('/home/sa0102/computer_vision/project/images/a.jpg') # queryImage
img2 = cv.imread('/home/sa0102/computer_vision/project/images/b.jpg') # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# Create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors, cross check in above command already applies ratio test
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 15 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:15],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
# Plt shows in BGR

#To-do: Implement Ransac


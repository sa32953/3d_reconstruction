#!/usr/bin/env python

########################################
#           Camera Parameters          #
########################################

import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import glob
import PIL.ExifTags
import PIL.Image
import time

s_t=time.clock()
# Define size of chessboard pattern to be found
# Easy mistake: wrong counting of pattern > count inside intersections of balck squares
chessboard_size = (9,6)

# Define arrays to save detected points
obj_points = [] #3D points in real world space 
img_points = [] #3D points in image plane

#Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)

objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

#Read Images with glob set
# TO BE CHANGED LATER ................
cal_images= glob.glob('/home/sa0102/computer_vision/project/images/calib_t2/samsung/*.jpg')

succ_count = 0

#Iterate for all images in specified path
for fname in cal_images:
    
    # Read image
    img= cv.imread(fname)

    # Convert to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find corners with OpenCV library func
    ret, corners = cv.findChessboardCorners(gray, chessboard_size , cv.CALIB_CB_FILTER_QUADS, cv.CALIB_CB_NORMALIZE_IMAGE)
    #print(ret)

    #Criteria for refining found points with subpixel func
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 150, 0.001)

    # If found, add object points, image points (after refining them)
    if ret == True:
        succ_count = succ_count + 1

        obj_points.append(objp)
        
        #Refine with subpixel func
        corners2 = cv.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
        img_points.append(corners)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboard_size , corners2, ret)

        img = cv.resize(img, None, fx=0.25, fy=0.25, interpolation= cv.INTER_CUBIC)
        #cv.imshow('img', img)
        #cv.waitKey(0)
        
#cv.destroyAllWindows()

# Calibrate Camera with computed points
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

#Save parameters into numpy file
np.save("/home/sa0102/computer_vision/project/source code/calibration/ret", ret)
np.save("/home/sa0102/computer_vision/project/source code/calibration/K", K)
np.save("/home/sa0102/computer_vision/project/source code/calibration/dist", dist)
np.save("/home/sa0102/computer_vision/project/source code/calibration/rvecs", rvecs)
np.save("/home/sa0102/computer_vision/project/source code/calibration/tvecs", tvecs)

# Find mean error of re-projection to see how much accuracy we have achieved
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(obj_points)))

print('Successfully found pattern in {} images'.format(succ_count))
print('Time taken to process: {} sec'.format(time.clock()-s_t))

# TO-do: 
# How to store focal length?
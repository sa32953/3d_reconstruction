#!/usr/bin/env python
import numpy as np 
import cv2 as cv 

filename = '/home/sa0102/computer_vision/project/images/a.jpg'

img = cv.imread(filename)

res=cv.resize(img, None, fx=0.25, fy=0.25, interpolation= cv.INTER_CUBIC)

gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
#print(gray.size)

#input arguments: src(should be a grayscale adn a float 32 type), blocksize, ksize, parameter
dst = cv.cornerHarris(gray,4,15,0.04)

# Threshold for an optimal value, it may vary depending on the image.
res[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',res)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
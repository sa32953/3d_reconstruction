#!/usr/bin/env python 3.6


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

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

h,  w = d_img1.shape[:2] # h is x-axis (4128), w is y-axis (3096)
new_cameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

#img1 = d_img1
#img2 = d_img2
img1 = cv.undistort(d_img1, K, dist, None, new_cameramtx)
img1 = img1[200:3900, 200:2900]
img2 = cv.undistort(d_img2, K, dist, None, new_cameramtx)
img2 = img2[200:3900, 200:2900]
# Cropping the bad part out
h,  w = img1.shape[:2] # h is x-axis (4128), w is y-axis (3096)
#print(h)
#print(w)

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

#E, mask = cv.findEssentialMat(pts1,pts2, K ,cv.FM_LMEDS)

# F prjects points from query image to train image ie from pts1 to pt2

# We select only inlier points
#pts1 = pts1[mask.ravel()==1]
#pts2 = pts2[mask.ravel()==1]

# Next step is to calculate normal vector of 2 projecting points and check if F transform 1 pt to other?
# Essential Matrix contains the information about translation and rotation...
# ...which describe the location of the second camera relative to the first in global coordinates

E = K.T.dot(F).dot(K)

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


R = U.dot(W).dot(Vt)
T = U[:, 2]

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

#print(T)
T_r = T.reshape((T.shape[0],-1))
#print(T_r)
#print(R)

zero = np.float32([[0.0], [0.0], [0.0]])
lastrow = np.float32([[0.0], [0.0], [0.0], [1]])
lastrow = lastrow.T

P1 = np.eye(4)
Kstar = np.hstack((K,zero))
Kstar = np.vstack((Kstar,lastrow))

RT = np.hstack((R,T_r))
RT = np.vstack((RT,lastrow))
#print(RT)

P2=np.dot(Kstar,RT)
#print(P2)

pts1T = pts1.transpose() #query
pts2T = pts2.transpose() #train
#print(len(mask))

# Find Homography

M, mask2 = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
# with homography points are found to be good (pt1 from query image is matching in img2, some
# features found are wrong through), take all points of img 1 and reporject (if pixels lie in range)
# then triangulate these set of points to get depth and plot 3d
# doubt: scaling of depth which i got from tringulation

allpts1 = []
for i in range(h):
    for j in range(w):
        allpts1.append([i,j])

allpts1 = np.float32(allpts1).reshape(-1,1,2)

# Transform
allpts2 = cv.perspectiveTransform(allpts1, M)

allpts1 = allpts1.T
allpts2 = allpts2.T
print(allpts1[0])
print(pts1[:5])

#pts = np.float32(pts1).reshape(-1,1,2)
#print(type(pts1))
#dst = cv.perspectiveTransform(pts,M)
#print(pts1[0])
#print(dst)
#print(pts2[0])
#dstimg = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
#dstimg = cv.resize(dstimg, None, fx=0.2, fy=0.2, interpolation= cv.INTER_CUBIC)
#cv.imshow('dstimg', dstimg)
#cv.waitKey(0)



#floatpst1T = np.array([pts1T[0,:].astype(np.float) / 4128.0, pts1T[1,:].astype(np.float) / 3096.0])
#floatpst2T = np.array([pts2T[0,:].astype(np.float) / 4128.0, pts2T[1,:].astype(np.float) / 3096.0])
#floatpst1T = np.array([pts1T[0,:].astype(np.float), pts1T[1,:].astype(np.float)])
#floatpst2T = np.array([pts2T[0,:].astype(np.float), pts2T[1,:].astype(np.float)])
floatpst1T = np.array([allpts1[0,:].astype(np.float), allpts1[1,:].astype(np.float)])
floatpst2T = np.array([allpts2[0,:].astype(np.float), allpts2[1,:].astype(np.float)])
print(floatpst1T)

#Triangulate
X = cv.triangulatePoints(P1[:3] ,P2[:3], floatpst1T, floatpst2T)

X = X/X[3]
#print(X.shape)
X = X[:3,:]

#cv.stereoRectify(,)

#print(floatpst1T[0])
#print(X)

#depth = K.dot(X)
#print(depth)

#cv.triangulatePoints()

# plot with matplotlib
Xs = X[0].T
Ys = X[1].T 
Zs = X[2].T

#print(Xs.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xs, Ys, Zs, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D point cloud: Use pan axes button below to inspect')
#plt.show()






for i in range(len(first_3d_point)):
    if first_3d_point[i][2]>0:
        pass
    else:
        #print('False')
        break

for i in range(len(first_3d_point)):
    if second_3d_point[i][2]>0:
        pass
    else:
        #print('False')
        break
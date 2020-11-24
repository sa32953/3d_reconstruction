


import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from pypcd import pypcd

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
x, y, w, h = roi
img1 = img1[y:y+h, x:x+w]
img2 = img2[y:y+h, x:x+w]
img1 = cv.resize(img1, None, fx=0.25, fy=0.25, interpolation= cv.INTER_CUBIC)
img2 = cv.resize(img2, None, fx=0.25, fy=0.25, interpolation= cv.INTER_CUBIC)
h,  w = img1.shape[:2]
########################################
#    Feature Description and Matching  #
########################################

# Initiate ORB detector
orb = cv.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#Draw feature points
s_img1 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
#plt.imshow(s_img1), plt.show()

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
F, mask = cv.findFundamentalMat(pts1,pts2, cv.FM_RANSAC, 1000, 0.999)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Next step is to calculate normal vector of 2 projecting points and check if F transform 1 pt to other?
r = np.linalg.matrix_rank(F)
print('Rank of F matrix {}'.format(r))

Estar, m = cv.findEssentialMat(pts1, pts2, new_cameramtx, cv.FM_LMEDS)
r3 = np.linalg.matrix_rank(Estar)
print('Rank of E* matrix {}'.format(r3))


########################################
#               SVD Matrix             #
########################################

U, S, Vt = np.linalg.svd(Estar)
#getting equal singular values
# also F matrix satisfies Epipolar eq

W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
R = U.dot(np.linalg.inv(W)).dot(Vt)
#R = U.dot(W).dot(Vt) #2nd case
#det of R ==1
print('Rotation found')

#from Elser Notes
t1 = S[1]*U[1][0]*U[2][1] - S[0]*U[1][1]*U[2][0]
t2 = -S[0]*U[0][0]*U[2][1] - S[0]*U[0][1]*U[2][0]
t3 = S[0]*U[0][0]*U[1][1] - S[0]*U[0][1]*U[1][0]

Tn = np.array([t1,t2,t3])
Tn = Tn / np.linalg.norm(Tn)
#Tn = -Tn
print('Transaltion found')

########################################
#               Check SVD              #
########################################

K_inv = np.linalg.inv(new_cameramtx)

first_inliers = []
second_inliers = []
for i in range(len(pts1)):
        # normalize and homogenize the image coordinates
        first_inliers.append([pts1[i][0], pts1[i][1], 1.0])
        second_inliers.append([pts2[i][0], pts2[i][1], 1.0])
first_inliers = np.array(first_inliers)
second_inliers = np.array(second_inliers)

# check if the point correspondences are in front of both images
def in_front_of_both_cameras(first_points, second_points, rot, trans):
    secondterm = (rot.T).dot(trans)
    firstterm = rot.T.dot(K_inv)

    for first, second in zip(first_inliers,second_inliers):
        first_3d_point = (firstterm.dot(first) - secondterm)
        second_3d_point = (firstterm.dot(second) - secondterm)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False
    return True

a = R
b = Tn
#first choice a = U * invW * Vt , Tn as per Elser notes
if not in_front_of_both_cameras(first_inliers, second_inliers, a, b):
    #print('Changed')
    # Second choice: a = same, b = - Tn
    b = - Tn
    if not in_front_of_both_cameras(first_inliers, second_inliers, a, b):
        #print('Changed 2wice')
        # 3rd choice: a = U * W * Vt, b = Tn
        b = Tn
        a = U.dot(W).dot(Vt)
        if not in_front_of_both_cameras(first_inliers, second_inliers, a, b):
            b = -Tn
            #print('Changed 3rice')
            # 4th choice: a = unchange, b = -Tn

R = a
Tn = b

# Helper vectors
zero = np.float32([[0.0], [0.0], [0.0]])
lastrow = np.float32([[0.0], [0.0], [0.0], [1]])
lastrow = lastrow.T

# Augment K
Kstar = np.hstack((K,zero))
Kstar = np.vstack((Kstar,lastrow))

# Augment RT
Tn = Tn.reshape((Tn.shape[0],-1))
RT = np.hstack((R,Tn))
RT = np.vstack((RT,lastrow))

# Camera Matrices
P1 = Kstar.dot(np.eye(4))
print('Cam Mat 1 found: Dim 4*4')
P2 = Kstar.dot(RT)
print('Cam Mat 2 found: Dim 4*4')
#print(P2)

########################################
#            Triangulation             #
########################################

pts1T = pts1.transpose() #query
pts2T = pts2.transpose() #train

floatpst1T = np.array([pts1T[0,:].astype(np.float), pts1T[1,:].astype(np.float)])
floatpst2T = np.array([pts2T[0,:].astype(np.float), pts2T[1,:].astype(np.float)])

X = cv.triangulatePoints(P1[:3] ,P2[:3], floatpst2T, floatpst1T)
X /= X[3] # homogenize

Y = X.T 

xs = []
ys = []
zs = []
# Y[0] is [-0.19105891 -0.36959654  0.56433563  1]
for i in range(len(Y)):
    xs.append(Y[i][0])
    ys.append(Y[i][1])
    zs.append(Y[i][2])

########################################
#                Plot                  #
########################################

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(xs, ys, zs, c='b', marker='o')
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.title('3D point cloud: Use pan axes button below to inspect')
#plt.show()

########################################
#                Export Ply            #
########################################

np.savetxt('pc.txt', Y,  delimiter= ' ')


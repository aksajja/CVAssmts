import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Stereo_Pair3.jpg',0)
w, h = int(img.shape[0]), int(img.shape[1]/2)

dst1 = img[:w][:,:h]
dst2 = img[:w][:,h:]

sift = cv2.xfeatures2d.SIFT_create()

#Compute the Keypoints and Descriptors with SIFT in left and right image
kp1, des1 = sift.detectAndCompute(dst1,None)
kp2, des2 = sift.detectAndCompute(dst2,None)

#Find the matching patches in both images by using FLANN matches
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []

#According to the lowe's paper, applying ratio test
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

#Converting list of points to numpy array
pts1 = np.array(pts1)
pts2 = np.array(pts2)

#Computate fundamental matrix by taking cv2.FM_LMEDS function()
F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

pts1 = pts1[:,:][mask.ravel()==1]
pts2 = pts2[:,:][mask.ravel()==1]
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
pts_new_1 = pts1.reshape((pts1.shape[0] * 2, 1))
pts_new_2 = pts2.reshape((pts2.shape[0] * 2, 1))

#Apply stereo rectifying function
retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(pts_new_1,pts_new_2,F,(100, 600))

#Apply wrapPerspetive function()
dst11 = cv2.warpPerspective(dst1,rectmat1,dst1.shape)
dst22 = cv2.warpPerspective(dst2,rectmat2,dst2.shape)
plt.imshow(dst11), plt.show()
plt.imshow(dst22), plt.show()

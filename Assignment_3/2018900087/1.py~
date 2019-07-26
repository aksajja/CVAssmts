import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Stereo_Pair3.jpg')    #read the input stereo image

w, h = int(img.shape[0]), int(img.shape[1]/2)

#Divide the input image into two partss
left_img = img[:w][:,:h]
right_img = img[:w][:,h:]

#RGB to Gray image
gray1= cv2.cvtColor(left_img ,cv2.COLOR_BGR2GRAY)	
gray2= cv2.cvtColor(right_img ,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()    #SIFT object creation
window_size = 15

#Find the keypoints in left image and right image
kp1 = [cv2.KeyPoint(x, y, window_size) for y in range(0, gray1.shape[0], window_size) for x in range(0, gray1.shape[1], window_size)]
kp2 = [cv2.KeyPoint(x, y, window_size) for y in range(0, gray2.shape[0], window_size) for x in range(0, gray2.shape[1], window_size)]

#Draw the Keypoints
img1=cv2.drawKeypoints(gray1, kp1, left_img)
img2=cv2.drawKeypoints(gray2, kp2, right_img)

_, dense_feat1 = sift.compute(gray1, kp1)
_, dense_feat2 = sift.compute(gray2, kp2)

#Find the matching points in both images
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(dense_feat1, dense_feat2)
matches = sorted(matches, key = lambda x:x.distance)

#Drawing the matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)

plt.imshow(img3), plt.show()

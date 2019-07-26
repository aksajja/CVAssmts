import cv2

img = cv2.imread('StereoImages/Stereo_Pair1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

dense=cv2.FeatureDetector_create("SIFT")
kp=dense.detect(imgGray)
kp,des=sift.compute(imgGray,kp)

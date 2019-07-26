import cv2
import math
import numpy as np
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt

u = cv2.imread('Stereo_Pair12.jpg')
w, h = int(u.shape[0]), int(u.shape[1]/2)

left_1 = u[:w][:,:h]
right_1= u[:w][:,h:]

left = cv2.cvtColor(left_1 ,cv2.COLOR_BGR2GRAY)
right = cv2.cvtColor(right_1 ,cv2.COLOR_BGR2GRAY)


temp1=0
temp2=0
count=0

img1 = []
img2 = []

w, h = int(left.shape[0] / 20), int(left.shape[1] / 20)

patches = image.extract_patches_2d(right, (20, 20))

print(w*h)

for i in range(h):
    for j in range(w):
        count = count+1
        sample = left[temp1:temp1+20][:,temp2:temp2+20]
        temp2 = temp2+20
        if(sample.shape == (20,20)): 
            img1.append(sample)  
            min_d = 9999999   
            for pat in patches:       
                t1 = np.sum(np.dot(sample,pat))
                t2 = math.sqrt(np.sum(sample**2))
                t3 = math.sqrt(np.sum(pat**2))
                corr = t1 / (t2+t3)
                if(corr < min_d):
                    min_d = corr
                    corr_p = pat
            img2.append(corr_p)
    print(sample, corr_p)  
            
    temp2 = 0
    temp1 = temp1 + 20 

sift = cv2.xfeatures2d.SIFT_create()

kp1 = sift.detect(left,None)
kp2 = sift.detect(right,None)


bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(img1, img2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)

plt.imshow(img3), plt.show()

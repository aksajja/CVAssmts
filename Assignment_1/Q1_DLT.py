import numpy as np
import cv2 as cv
from scipy import linalg
from numpy.linalg import inv

def findImagePoints():
    img = cv.imread('surya.JPG',0)
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=100,param2=30,minRadius=10,maxRadius=80)
    circles = np.uint16(np.around(circles))
    imgp = []
    for i in circles[0,:]:
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        imgp.append(i[0],i[1])

    imgp = np.array(imgp)
    return imgp
    
def DLT(objp, imgp):

    mat = np.zeros((12,12), np.int)
    for i in range(0,12):
        if(i%2==0):
            mat[i][0:3] = np.multiply(objp[int(i/2)][0:3], -1)
            mat[i][3] = -1
            x = imgp[int(i/2)][0]
            mat[i][8:11] = np.multiply(objp[int(i/2)][0:3],x)        
            mat[i][11] = x
        else:
            mat[i][4:7] = np.multiply(objp[int(i/2)][0:3], -1)
            mat[i][7] = -1
            y = imgp[int(i/2)][1]
            mat[i][8:11] = np.multiply(objp[int(i/2)][0:3],y) 
            mat[i][11] = y

    U, s, V_h = linalg.svd(mat)
    
    proj = np.reshape(V_h[-1], (3, 4))
    
    print(proj)
    
    K, R = np.linalg.qr(proj)

    #P=K[R|t]
    print(R)
    h = R.T[-1]
    
    print(R.T)
    
    K_inv = inv(K)

    t = np.matmul(K_inv, h)

    R_temp = np.delete(R, -1, axis=1)
    R_temp = np.insert(R_temp, 3, t, axis=1)
    return K, R, t

#imgp = findImagePoints()

objp = np.array([[36,36,0,1], [0,36,0,1],[0,72,0,1],[36,0,36,1], [0,0,36,1], [0,0,72,1]])
imgp = np.array([[4037,1356], [4886,1422], [4931, 583], [3883, 2404], [4743, 2481], [4655, 2812]])

K, R, t = DLT(objp,imgp)

print("Camera Matrix: ", K)
print("Rotation Matrix: ", R)
print("Translation Vector:", t)

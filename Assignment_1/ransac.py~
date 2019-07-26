import numpy as np
import random
import cv2 as cv
from scipy import linalg
from numpy.linalg import inv

def check(maybeInliers,objp):
    return any(np.array_equal(x, maybeInliers) for x in objp)

def RANSAC(objp, imgp):
    mat = np.zeros((2*objp.shape[0],12), np.int)
    for i in range(2*objp.shape[0]):
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
    return proj
    
def ErrorCalc(proj, objp, imgp):
    inliers = []
    inliers1 = []
    Error = []
    for i in range(36):
        verify = np.matmul(proj, objp[i])
        verify = [(verify[0]/verify[2]), (verify[1]/verify[2])]
        Error.append(np.average(np.array(imgp[i]) - verify))
        if((np.average(np.array(imgp[i]) - verify)) <= 10.0):
            inliers.append(imgp[i])
            inliers1.append(objp[i])
    return(Error, inliers, inliers1)
    
objp=np.array([(0,36,0,1),(36,36,0,1),(72,36,0,1),(108,36,0,1),(144,36,0,1),(180,36,0,1),(216,36,0,1),(0,72,0,1),(36,72,0,1),(72,72,0,1),(108,72,0,1),(144,72,0,1),(180,72,0,1),(216,36,0,1),(0,0,36,1),(36,0,36,1),(72,0,36,1),(108,0,36,1),(144,0,36,1),(180,0,36,1),(0,0,72,1),(36,0,72,1),(72,0,72,1),(108,0,72,1),(144,0,72,1),(180,0,72,1),(0,0,108,1),(36,0,108,1),(72,0,108,1),(108,0,108,1),(144,0,108,1),(0,0,144,1),(36,0,144,1),(72,0,144,1),(108,0,144,1),(144,0,144,1)])

imgp=np.array([(4876,1404),(4037,1338),(3221,1283),(248,1228),(1676,1173),(926,1117),(198,1062),(4930,566),(4059,533),(3243,478),(2438,433),(1654,378),(893,334),(132,290),(4732,2466),(3868,2383),(3031,2306),(2224,2236),(1431,2166),(658,2108),(4648,2792),(3727,2722),(2845,2639),(1988,2549),(1150,2466),(331,2409),(4546,3176),(3574,3093),(2627,3003),(1719,2914),(824,2824),(4437,3617),(3388,3509),(2378,3413),(1412,3323),(459,3227)])

max1 = 9999999

for i in range(3):
    imgp = imgp.astype(float)
    pairs = list(zip(imgp, objp))
    pairs = random.sample(pairs, 6)
    A1, B1 = zip(*pairs)
    imgp_temp=list(A1)
    objp_temp=list(B1)
    objp_temp = np.array(objp_temp)
    imgp_temp = np.array(imgp_temp)
    proj_temp = RANSAC(objp_temp, imgp_temp)
    Error, inliers, inliers1 = ErrorCalc(proj_temp, objp, imgp)
    if(len(inliers)>=6):
        proj_temp = RANSAC(objp_temp, imgp_temp)
        Error, inliers, inliers1 = ErrorCalc(proj_temp, objp, imgp)
        if(np.average(Error)< max1):
            max1 = np.average(Error)
            proj = proj_temp

print("Final Projection Matrix: ", proj)
K, R = np.linalg.qr(proj)
h = R.T[-1]
K_inv = inv(K)
t = np.matmul(K_inv, h)
print(K, R, t)

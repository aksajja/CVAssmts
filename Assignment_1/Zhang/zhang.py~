import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((8*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('*.JPG')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)
    print(corners)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        cv.drawChessboardCorners(img, (8,6), corners, ret)
        image = cv.resize(img, (800,800))
        cv.imshow('img', image)
        cv.waitKey(500)


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix: ", mtx)
print("Rotation Matrix: ", rvecs)
print("Translation Vector:", tvecs)

cv.destroyAllWindows()

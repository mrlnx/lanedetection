#!/usr/bin/python

import cv2
import glob
import numpy as np
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob("images/*jpg")

for filename in images:
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#print filename	

	ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
	
	if ret == True:

		objpoints.append(objp)

		corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
		imgpoints.append(corners2)		
	
		image = cv2.drawChessboardCorners(image, (7, 6), corners2, ret)


		#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, gray.shape[::-1], None, None)

		#cv2.imshow('Image', image)
		#cv2.waitKey(2000)

		print filename + " found!"
	#else:
		#os.remove(filename)			#print filename + " remove!"


print imgpoints
print objpoints

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640, 48),  None, None)

print "mtx: "
print mtx

print "dist: "
print dist

cv2.destroyAllWindows()

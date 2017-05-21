#[[ 723.79767097    0.          321.53022265]
# [   0.          725.70162004  200.53503688]
# [   0.            0.            1.        ]] 


#[[ 0.12311024  0.08672342 -0.01571489  0.00133201 -0.90823159]]


import numpy as np
import cv2

mCamera = np.matrix([[723.79767097, 0., 321.53022265], [0., 725.70162004, 200.53503688], [0., 0., 1.]])

vDistCoeff = np.array([0.12311024, 0.08672342, -0.01571489, 0.00133201, -0.90823159])

print mCamera
print vDistCoeff

try:

	video = "";

	exp = cv2.undistort(video, mCamera, vDistCoeff)


	print exp

except TypeError:
	print "Type error!"
	

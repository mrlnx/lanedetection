import cv2
import numpy as np
import time
import pickle
import os

mCamera = np.matrix([[723.79767097, 0., 321.53022265], [0., 725.70162004, 200.53503688], [0., 0., 1.]])

vDistCoeff = np.array([0.12311024, 0.08672342, -0.01571489, 0.00133201, -0.90823159])

print mCamera
print vDistCoeff

capture = cv2.VideoCapture(0)

while(capture.isOpened()):

        ret, frame = capture.read()
	frame = cv2.flip(frame, 1)

	undist = cv2.undistort(frame, mCamera, vDistCoeff)

	cv2.imshow("Frame", undist)
	#cv2.imshow("Frame2", frame)

	if cv2.waitKey(1) == 27:
		break

capture.release()
cv2.destroyAllWindows()




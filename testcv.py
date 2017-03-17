import cv2
import numpy
import matplotlib
import time

capture = cv2.VideoCapture(0)

while capture.isOpened():
	ret, img = capture.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cv2.imshow("img", gray)

	key = cv2.waitKey(30) & 0xff

	if key == 27:
		break

capture.release()
cv2.destroyAllWindows()

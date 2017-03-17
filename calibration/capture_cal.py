#! /usr/bin/python

import cv2
import numpy as np
import time
import pickle
import os

capture = cv2.VideoCapture(0)

#persistence filename
p_name = "counter"

i = 2800

#if os.path.exists(p_name):
#	fileObj = open(p_name, "r")
#	s = pickle.load('counter')

#	i = s[0]
#	print "Open file " + p_name
#else:

#	print "Create file " + p_name

#	fileObj = open(p_name, "wb")
#	pickle.dump(1, fileObj)
	
while(capture.isOpened()):
	
	ret, frame = capture.read()

	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	gray = cv2.flip(frame, 1)

	#cv2.imshow('Frame gray', gray)
	#cv2.imshow('Frame color', frame)

	ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)


	#if cv2.waitKey(1) & 0xFF == ord("q"):
	if ret == True:
		cv2.imshow('Frame color', frame)
		cv2.imwrite('images/calibration' + str(i) + '.jpg', gray)
		i = i + 1

print "last count " + i
	
capture.release()
cv2.destroyAllWindows()

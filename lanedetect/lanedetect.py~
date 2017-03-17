#!/usr/bin/python

import cv2
import numpy as np

def initVideo(file):

	videoPath = file
	capture = cv2.VideoCapture(videoPath)

	return capture

def gray(frame):
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def gaussianBlur(frame, size):
	return cv2.GaussianBlur(frame, (size, size), 0)

def cannyEdge(frame, low, high):
	return cv2.Canny(frame, low, high)

#def threshold(frame):

def region(frame, vertex):
	mask = np.zeros_like(frame)
	
	if len(frame.shape) > 2:
		channel_count = frame.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	cv2.fillPoly(mask, vertex, ignore_mask_color)

	return cv2.bitwise_and(frame, mask)

def computeROI(frame):
        max_width = frame.shape[1]
        max_height = frame.shape[0]
        width_delta = int(max_width/20)

	bottom_left = (100, max_height)
	bottom_right = (max_width - 100, max_height)
	top_right = (max_width / 2 + width_delta, max_height / 2 + 50)
	top_left = (max_width / 2 - width_delta, max_height / 2 + 50)

        return np.array([[bottom_left, bottom_right, top_right, top_left]], np.int32)


def zeros(frame):
	return np.zeros(frame.shape, dtype=np.uint8)


def houghLines(frame):

	rho = 3
	theta = np.pi / 180
	thres = 50
	min_line_length = 180
	max_line_gap = 80

	return cv2.HoughLinesP(frame, rho, theta, thres, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

def drawLines(frame, original):

	lines = houghLines(frame)
	lines_original = zeros(original)
	line_width = 2

	color = (0, 0, 255)
		
	for line in lines:
	
		if line != None:

			for x1, y1, x2, y2 in line:
			
				#print x1
				#print y1
				#print x2
				#print y2

				cv2.line(original, (x1, y1), (x2, y2), color, line_width)
	
	

def manipulateFrame(frame):

	# variables
	kernel_size = 7
	low_thres = 50
	high_thres = 125

	# keep original image
	original = frame

	# frame gray
	frame = gray(frame)

	# blur
	frame = gaussianBlur(frame, kernel_size)
	
	# canny
	frame = cannyEdge(frame, low_thres, high_thres)

	# draw lines
	#frame = drawLines(frame, original)

	# compute regio of interest
	vertex = computeROI(frame)

	# set regio
	frame = region(frame, vertex)
	
	drawLines(frame, original)

	return original

def playVideo(capture):
	while(capture.isOpened()):
		ret, frame = capture.read()
		man = manipulateFrame(frame)

		cv2.imshow("Video", man)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	capture.release()
	cv2.destroyAllWindows()

filePath = "video/project_video.mp4"
video = initVideo(filePath)
playVideo(video)

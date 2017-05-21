#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt

mCamera = np.matrix([[723.79767097, 0., 321.53022265], [0., 725.70162004, 200.53503688], [0., 0., 1.]])
vDistCoeff = np.array([0.12311024, 0.08672342, -0.01571489, 0.00133201, -0.90823159])

def initVideo(file):

	videoPath = file
	capture = cv2.VideoCapture(videoPath)
	
	#catpure = cv2.resize(capture, (640, 480))

	return capture

def white_yellow(frame):

	conv = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
	
	# white mask
	lower = np.uint8([0, 200, 0])
	upper = np.uint8([255, 255, 255])
	white_mask = cv2.inRange(conv, lower, upper)

	# yellow mask
	lower = np.uint8([10, 0, 100])
	upper = np.uint8([40, 255, 255])
	yellow_mask = cv2.inRange(conv, lower, upper)

	# set masks
	mask = cv2.bitwise_or(white_mask, yellow_mask)
	return cv2.bitwise_and(frame, frame, mask = mask)

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

	#print vertex[0]
	#print vertex[0][1]
	#print vertex[1][0]
	#print vertex[1][1]

	cv2.fillPoly(mask, vertex, ignore_mask_color)

	return cv2.bitwise_and(frame, mask)

def computeROI(frame, top_width, bottom_width):
      
	top_width = top_width / 2
	bottom_width = bottom_width / 2

	max_width = frame.shape[1]
        max_height = frame.shape[0]
        width_delta = int(max_width/20)

	bottom_left = (bottom_width, max_height)
	bottom_right = (max_width - bottom_width, max_height)
	top_right = (max_width / 2 + width_delta, max_height / 2 + top_width)
	top_left = (max_width / 2 - width_delta, max_height / 2 + top_width)

        nparray = np.array([[bottom_left, bottom_right, top_right, top_left]], np.int32)
	
	cv2.polylines(frame, nparray, 1, 50)

	return nparray

def zeros(frame):
	return np.zeros(frame.shape, dtype=np.uint8)


def houghLines(frame):

	rho = 2
	theta = np.pi / 180
	thres = 120
	min_line_length = 20
	max_line_gap = 20

	hough = cv2.HoughLinesP(frame, rho, theta, thres, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

	if(hough is None or len(hough) > 500):
		return []
	else:
		return hough


def tranform(undist, src, dst):

	pers = cv2.getPerspectiveTransform(src, dst)
	warp = cv2.warpPerspective(undist, pers, (640, 480))

	return warp

def drawLines(frame, original):

	lines = houghLines(frame)
	lines_original = zeros(original)
	line_width = 2

	color = (0, 0, 255)

	#print lines

	for line in lines:
		#print line

		try:		
			for x1, y1, x2, y2 in line:
				
				if x2 == x1:
					continue

				current_slope = (y2 - y1) / (x2 - x1)
				intercept = y1 - current_slope * x1				
						
				#print current_slope

				x_min = x1
				y_min = y1
				x_max = x2
				y_max = y2

				cv2.line(original, (x_min, y_min), (x_max, y_max), color, line_width)


		except TypeError:
			print "Type Error"

def manipulateFrame(frame):

	# variables
	kernel_size = 7
	low_thres = 75#50
	high_thres = 100#125

	# keep original image
	original = frame

	#frame = white_yellow(original)

	# frame gray
	frame = gray(frame)

	# blur
	frame = gaussianBlur(frame, kernel_size)
	
	# canny
	frame = cannyEdge(frame, low_thres, high_thres)
	edges = frame
	color_edges = np.dstack((edges, edges, edges));

	#frame = cv2.addWeighted(color_edges, 0.8, original, 1, 0)

	# draw lines
	frame = drawLines(frame, original)

	# compute regio of interest
	vertex = computeROI(original, 100, 50)

	# set regio
	frame = region(original, vertex)

	#drawLines(frame, original)

        #original = cv2.addWeighted(color_edges, 0.8, original, 1, 0)


	return original

def playVideo(capture):
	while(capture.isOpened()):

		#mera = np.matrix([[723.79767097, 0., 321.53022265], [0., 725.70162004, 200.53503688], [0., 0., 1.]])

		#vDistCoeff = np.array([0.12311024, 0.08672342, -0.01571489, 0.00133201, -0.90823159])


		ret, frame = capture.read()
		frame = cv2.resize(frame, (640, 480))
		
		#src = np.float32([[20, 20], [], [], []])
		#dst = np.float32([[], [], [], []])	

	#	ipm_dst = computeROI(frame, 100, 50)
	#	ipm_src = computeROI(frame, 100, 50)
		#cv2.polylines(frame, [ipm_dst], True, (0,0,255))

		
		man = manipulateFrame(frame)
		#trans = tranform(frame, ipm_src, ipm_src)

		#undist = cv2.undistort(frame, mCamera, vDistCoeff)

		#cv2.imshow("Undist video", undist)
		#cv2.imshow("Video", frame)
		#cv2.imshow("Warped", trans)
		cv2.imshow("Manipulated", man)

	#	plt.imshow(man)
	#	plt.show()

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	capture.release()
	cv2.destroyAllWindows()

filePath = "video/project_video.mp4"
video = initVideo(filePath)
playVideo(video)

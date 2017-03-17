import cv2
import urllib
import numpy as np

stream = urllib.urlopen("http://192.168.0.27:8090/camera.mjpeg")

fourcc = cv2.VideoWriter_fourcc(*'jpeg')
out = cv2.VideoWriter('~/Documents/video/output/webcam.mov', fourcc, 20.0, (320, 240))

bytes = bytes()

while True:

	bytes += stream.read(1024)
	
	a = bytes.find('\xff\xd8')
	b = bytes.find('\xff\xd9')

	if a != -1 and b != -1:
		jpg = bytes[a:b+2]
		bytes = bytes[b+2:]

		image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
		
		if image is not None:
			cv2.imshow('image', image)

			out.write(image)

			# press q to quit
			if cv2.waitKey(1) == 27:
				exit(0)

cv2.destroyAllWindows()

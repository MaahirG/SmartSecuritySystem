# USAGE
# python motion_detector.py

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# reading from webcam
vs = VideoStream(src=0).start()
time.sleep(1.0)

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied text
	frame = vs.read()
	text = "Unoccupied"
	
	# if the frame could not be grabbed, then we have reached the end of the video
	if frame is None:
		break

	haarFrame = frame
	grayHaar = cv2.cvtColor(haarFrame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(grayHaar, 1.3, 5)         # tuning params based on size
	
	for (x,y,w,h) in faces: # only enters loop if non empty
		cv2.rectangle(haarFrame, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_gray = grayHaar[y:y+h, x:x+w]                           # create a smaller region inside faces to detect eyes --> won't find eye outside of face
		roi_color = haarFrame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)
		
	
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
	# if the first frame is None, initialize it
	if firstFrame is None:
		print(frame)
		firstFrame = gray
		continue

	# Object Detection/Edge Detection
	# compute the absolute elementwise (two arrays) difference between the current frame and first frame
	frameDelta = cv2.absdiff(firstFrame, gray) # new array of differences in pixels
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1] # new array of thresholded (camera inaccuracies) differences in pixels returned as thresh
	

	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)  # dilation fills in neighbouring pixels as 1 (for binary) to add pixels to the outer border (dilating pixels in the threshold array)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	# CHAIN_APPROX_SIMPLE takes away redundancies when highlighting the curves/joining parts of a threshold frameDelta
	# cnts is a tuple - should have 2 things inside - a vector of vectors all separate contours (with xy vector of coords defining contour) and a vector of hierarchies
	# once we grab contours (tuple of relevant x,y coords highlighting the curves joining a shape)
	print(type(cnts))
	print(cnts)
	cnts = imutils.grab_contours(cnts) # grabs only the x,y coord tuple

	# loop over the contours - loop through lists inside tuple containing respective contour x,y coords (depending on how many objects were detected separate) - if 1 object, one list in the tuple
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 500:  	# 500 default, area based on boundary coordinates from contour
			continue								# disregard but keep looping the rest of the contours
		
		# compute the bounding box for the contour, draw it on the frame, and update the text
		(x, y, w, h) = cv2.boundingRect(c)								# gets rectangle dimensions based on entire list of contour coords
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)


	cv2.imshow('Haar Face Detection Algorithm', haarFrame)


	# show the frame and record if the user presses a key
	cv2.imshow("Documented Background", firstFrame)
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Threshold", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()
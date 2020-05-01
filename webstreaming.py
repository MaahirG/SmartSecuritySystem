# python webstreaming.py --ip 192.168.0.20 --port 8000

# Tunable parameters: 
# min area for contour, dilation iterations, threshold for pixel density difference
# cascade params, image resize and corresponding gaussian blur, accumulated weight avg


from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2


# initialize the output frame and a lock used to ensure thread-safe exchanges of the output frames (useful for multiple browsers/tabs are viewing tthe stream)
outputFrame = None			# Frame post motion pushed to server
lock = threading.Lock()		# for concurrency multi user


# initialize a flask object
app = Flask(__name__)


# initialize the video stream and allow the camera to warmup
video = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


def detect():
	# grab global references to the video stream, output frame, and lock variables
	global video, outputFrame, lock

	#motionObj = MotionFacialDetector()

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	# initialize the first frame in the video stream
	firstFrame = None
	avg = None
	# loop over the frames of the video
	while True:
		# grab the current frame and initialize the occupied/unoccupied text
		frame = video.read()
		text = "Unoccupied"
		frame = imutils.resize(frame, width=625)

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
			
		
		# Blur the resized and grayscaled frame
		gray = cv2.GaussianBlur(grayHaar, (25, 25), 0)
		
		# if the first frame is None, initialize it
		if firstFrame is None:
			firstFrame = gray
			continue

		# if the average frame is None, initialize it
		if avg is None:
			avg = gray.copy().astype("float")
			continue

		# Object Detection/Edge Detection

		# accumulate the weighted average between the current frame and (accumulated)previous frames, then compute the difference between the current frame and running average
		cv2.accumulateWeighted(firstFrame, avg, 0.5)	# avg is accumulated slight differences in pixels over the iteration of frames, 0.5 is weight for gray frame
		frameDelta = cv2.absdiff(firstFrame, gray)

		# compute the absolute elementwise (two arrays) difference between the current frame and first frame
		# frameDelta = cv2.absdiff(firstFrame, gray) # new array of differences in pixels
		thresh = cv2.threshold(frameDelta, 55, 255, cv2.THRESH_BINARY)[1] # new array of thresholded (camera inaccuracies) differences in pixels returned as thresh
		

		# dilate the thresholded image to fill in holes, then find contours on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=6)  # dilation fills in neighbouring pixels as 1 (for binary) to add pixels to the outer border (dilating pixels in the threshold array)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	# CHAIN_APPROX_SIMPLE takes away redundancies when highlighting the curves/joining parts of a threshold frameDelta
		# cnts is a tuple - should have 2 things inside - a vector of vectors all separate contours (with xy vector of coords defining contour) and a vector of hierarchies
		# once we grab contours (tuple of relevant x,y coords highlighting the curves joining a shape)
		# print(type(cnts))
		# print(cnts)
		cnts = imutils.grab_contours(cnts) # grabs only the x,y coord tuple

		# loop over the contours - loop through lists inside tuple containing respective contour x,y coords (depending on how many objects were detected separate) - if 1 object, one list in the tuple
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < 800:  	# area based on boundary coordqinates from contour
				continue								# disregard but keep looping the rest of the contours
			
			# compute the bounding box for the contour, draw it on the frame, and update the text
			(x, y, w, h) = cv2.boundingRect(c)								# gets rectangle dimensions based on entire list of contour coords
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			text = "Occupied"

		# draw the text and timestamp on the frame
		cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)


		with lock:
			outputFrame = frame.copy()



def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')



@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media type (mime type)
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect)
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
video.stop()
# python motion_detector.py

# Tunable parameters: 
# min area for contour, dilation iterations, threshold for pixel density difference
# cascade params, image resize and corresponding gaussian blur, accumulated weight avg

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy

import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BOARD)    # pin #'s are gpio spots

# GPIO.setup(11, GPIO.OUT)
# servo = GPIO.PWM(11,50)      # 50 is pulse rate (Hz)

# servo.start(7)


face_cascade = cv2.CascadeClassifier('haarModels/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarModels/haarcascade_eye.xml')

# loading mobilenet ssd models into opencv dnn library
model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')


# Pretrained classes dictionary in the model
classDict = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


# for servo
def SetAngle(angle):
	duty = (angle / 18) + 2
	GPIO.output(11, True)
	servo.ChangeDutyCycle(duty)
	time.sleep(1)
	GPIO.output(11, False)
	servo.ChangeDutyCycle(0)

def idToClassName(class_id, classes):		#classes is a dictionary that holds all classnames
	for key, value in classes.items():
		if class_id == key:
			return value	# returns actual class name -- dog, cat, human etc instead of a class_id (number)

# need to change to RPI camera enabled.
# Start reading from webcam
vs = VideoStream(src=0).start()
time.sleep(1.0)

# initialize the first frame in the video stream
firstFrame = None
avg = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied text
	frame = vs.read()
	text = "Unoccupied"
	img = imutils.resize(frame, width=625)

	# blob is like a tensor, a compact object of data (preprocessed image). need to feed into the net
	# the specific model used relies on 300x300 input image for its architecture
	# model relies on RGB input not BGR like opencv (swapRB)
	model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True))
	
	# Network produces output blob with a shape 1x1xNx7 where N is a number of --> the numbers in shape are the sizes of each vector enclosed by that blob.
	# detections and every detection is a vector of values --> 7 is the size of each detection vector (7 prediction criterion)
	# [batchId, classId, confidence, left, top, right, bottom]
	output = model.forward()
	print(output[0,0,50,6])  # the 6th option --> bottom of the 50th prediction.. the first two have size 1 so 0 is the only index option
	image_height, image_width, _ = frame.shape
	print(frame.shape)
	print(output.shape)
	for detection in output[0,0,:,:]:
		confidence = detection[2]		# 2 because confidence is in the 2 spot of the detection vector
		
		if confidence > 0.5:
			classID = detection[1]
			className = idToClassName(classID, classDict)
			print(str(str(classID) + " " + str(detection[2])  + " " + className))
			
			box_x = detection[3] * image_width # detections are % left etc.
			box_y = detection[4] * image_height
			box_width = detection[5] * image_width
			box_height = detection[6] * image_height
			cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
			cv2.putText(frame, className, (int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))

			if className == 'person':
				text = "Occupied"

	haarFrame = img
	grayHaar = cv2.cvtColor(haarFrame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(grayHaar, 1.3, 5)         # tuning params based on size
	
	for (x,y,w,h) in faces: # only enters loop if non empty
		print("x is: ", x, " y is: ", y, " w is: ", w, " h is: ", h)

		# if(x > 300):
		# 	SetAngle(180)
		# elif(x<=300 and x > 240):
		# 	SetAngle(144)	
		# elif(x<=240 and x > 180):
		# 	SetAngle(108)
		# elif(x<=180 and x > 120):
		# 	SetAngle(72)
		# elif(x<=120 and x > 60):
		# 	SetAngle(36)
		# elif(x<=60 and x >= 0):
		# 	SetAngle(0)

		cv2.rectangle(haarFrame, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_gray = grayHaar[y:y+h, x:x+w]                           # create a smaller region inside faces to detect eyes --> won't find eye outside of face
		roi_color = haarFrame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)



	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

	cv2.imshow('Haar Face Detection Algorithm', haarFrame)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		servo.ChangeDutyCycle(2)
		time.sleep(1)
		servo.stop()
		GPIO.cleanup()
		print("BYE SERVO")
		break

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()

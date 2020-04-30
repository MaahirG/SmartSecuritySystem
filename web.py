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


class MotionFacialDetector:
    def motionDetector(firstFrame, gray):
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
        return cnts
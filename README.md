# Smart Security System

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## V3 Current
The current version relies on a convolutional neural network and takes into account the ROI of detection and moves a servo motor enabling the security camera to swivel. </br>
Model architecture: Mobilenet v2 SSD 
Training Set: COCO

## V2
The next version implemented haar cascades to conduct object detection. The haar model relied on a previously trained feature extracted data set which further relied on a degenerate decision tree that would either be 'yes' the feature exists now keep going and check the next feature, or feature doesn't exist --> not one of the trained objects.

## V1
The project started as a simple background segmentation based on adjacent pixel value thresholding, which was then used for a motion detector. This was debunked as shadows destroyed the model.

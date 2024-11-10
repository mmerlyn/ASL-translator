# USAGE
# python skindetector.py
# python skindetector.py --video asl_abc_2_crop.mp4

# import the necessary packages
import imutils
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([25, 255, 255], dtype = "uint8")

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	frame =  cv2.resize(frame, (224, 224))
	diff = cv2.absdiff(frame.astype("uint8"), frame)


	t = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
	cv2.imshow("T",t)
	edged = cv2.Canny(frame, 30,150)
	cv2.imshow("Edges Canny", edged)

    # get the contours in the thresholded image
	(cnts,_)=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	segmented = max(cnts, key=cv2.contourArea)
	cv2.drawContours(frame,segmented,-1,(0,255,0),2)
	cv2.imshow("ss",frame)








	#converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
	#skinMask = cv2.erode(skinMask, kernel, iterations = 1)
	#skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	#skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	#cv2.imshow("mask",skinMask)
	#skin = cv2.bitwise_and(frame, frame, mask = skinMask)

	# show the skin in the image along with the mask
	#cv2.imshow("images", np.hstack([frame, skin]))

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

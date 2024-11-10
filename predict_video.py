# USAGE
# python predict_video.py --model output/finalactivity3.model --label-bin output/finallb3.pickle --input my_abc.mp4 --output output/my_abc.avi --size 1

# import the necessary packages
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
#mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([25, 255, 255], dtype = "uint8")
fgbg = cv2.createBackgroundSubtractorMOG2()
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	output = frame.copy()
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	frame =  cv2.resize(frame, (224, 224))

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imshow("T",thresh1)








	#converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#skinMask = cv2.inRange(converted, lower, upper)

	## apply a series of erosions and dilations to the mask
	## using an elliptical kernel
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
	#skinMask = cv2.erode(skinMask, kernel, iterations = 1)
	#skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	## blur the mask to help remove noise, then apply the
	## mask to the frame
	#skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	#cv2.imshow('mask',skinMask)




	#frame = cv2.resize(frame, (224, 224))#.astype("float32")
	##frame -= mean
	#cv2.imshow('frame',frame)
	#blur = cv2.GaussianBlur(frame, (3,3), 0)
	#hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
	#lower_color = np.array([108, 23, 82])
	#upper_color = np.array([179, 255, 255])
	#mask = cv2.inRange(hsv, lower_color, upper_color)
	#cv2.imshow('mask',mask)
	#blur = cv2.medianBlur(mask, 5)
	#cv2.imshow('blur',blur)
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	#hsv_d = cv2.dilate(blur, kernel, iterations=3)
	#cv2.imshow('dialted input',hsv_d)

	thresh1 = np.expand_dims(thresh1, axis=-1)
	preds =model.predict(np.expand_dims(thresh1, axis=0))[0]
	Q.append(preds)

	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
	# draw the activity on the output frame
	text = "activity: {} ".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 5)
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)
	# write the output frame to disk
	writer.write(output)
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

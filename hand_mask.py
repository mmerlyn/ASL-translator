from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import time

from imutils.video import VideoStream

"""
python hand_mask.py --model output/finalactivity3.model --label-bin output/finallb3.pickle --output output/asl_my_demo.avi --size 1
"""

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")

ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

print("[INFO] starting video stream...")
(W, H) = (None, None)
Q = deque(maxlen=args["size"])
writer = None
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
vs = cv2.VideoCapture(0)
zzz = []
while(True):

	(g,frame) = vs.read()
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	output = frame.copy()
	#cv2.imshow('original',frame)
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224))#.astype("float32")
	#cv2.imshow('frame',frame)
	blur = cv2.GaussianBlur(frame, (3,3), 0)

	hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
	lower_color = np.array([108, 23, 82])#np.array([108, 23, 82])
	upper_color = np.array([179, 255, 255])#np.array([179, 255, 255])
	mask = cv2.inRange(hsv, lower_color, upper_color)
	cv2.imshow('mask',mask)
	blur = cv2.medianBlur(mask, 5)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	#mask = cv2.erode(mask,kernel, iterations=2)
	#cv2.imshow('eroded mask',mask)
	hsv_d = cv2.dilate(blur, kernel)
	cv2.imshow('dialted input',hsv_d)
	#hsv_d = cv2.GaussianBlur(blur, kernel)

	#cv2.imshow('befor blur', hsv_d)
	hsv_d = np.expand_dims(hsv_d, axis=-1)
	preds =model.predict(np.expand_dims(hsv_d, axis=0))[0]
	Q.append(preds)

	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
	zzz.append(label)
	print(zzz)
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)

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

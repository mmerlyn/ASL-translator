from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import time
import keras

from imutils.video import VideoStream

bg = None
aWeight = 0.5
num_frames = 0
top, right, bottom, left = 175,350,399,574
fgbg = cv2.createBackgroundSubtractorMOG2()
"""
python sign_predict.py --model output/finalactivity3.model --label-bin output/finallb3.pickle --output output/pbl.avi --size 1
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

# initialize the image mean for mean subtraction along with the
# predictions queue
L = []
Q = deque(maxlen=args["size"])

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
	global bg
	diff = cv2.absdiff(bg.astype("uint8"), image)
	fgim = fgbg.apply(image)
	masked = cv2.bitwise_and(image,image,mask=fgim)
	cv2.imshow("fg",masked)
	thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1] #25
	(cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) == 0:
		return
	else:
		segmented = max(cnts, key=cv2.contourArea)
		return (thresholded, segmented)




print("[INFO] starting video stream...")
vs =cv2.VideoCapture(0)# cv2.VideoCapture("asl_three_letter_word_2_crop.mp4")#cv2.VideoCapture(0)#cv2.VideoCapture("asl_abc_2_crop_2_black_crop.mp4")
#VideoStream(src=0).start()

writer = None
(W, H) = (None, None)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	(g,frame) = vs.read()
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	frame = cv2.flip(frame,1)
	output = frame.copy()
	roi = frame[top:bottom, right:left]
	gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #frame
	blur = cv2.GaussianBlur(gray,(7,7),0)
	thresholded = blur.copy()

	if num_frames <20:
		run_avg(gray, aWeight)
		fgim = fgbg.apply(gray)
		masked = cv2.bitwise_and(gray,gray,mask=fgim)
		#cv2.imshow("bg",masked)


	else:
		hand = segment(gray)
		if hand is not None:
			(thresholded, segmented) = hand
	cv2.rectangle(output, (left,top), (right,bottom),(0,255,0),2)

	thresholded = cv2.resize(thresholded, (224,224))
	cv2.imshow("Thresholded", thresholded)
	num_frames +=1

	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#frame = cv2.resize(frame, (224, 224)).astype("float32")
	#frame -= mean
	#cv2.imshow("Input", frame)
	thresholded = np.expand_dims(thresholded, axis=-1)
	preds = model.predict(np.expand_dims(thresholded, axis=0))[0]
	Q.append(preds)

	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
	L.append(label)
	text = "Symbol: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print(L)
		break

# release the file pointers

print("[INFO] cleaning up...")

writer.release()
vs.release()

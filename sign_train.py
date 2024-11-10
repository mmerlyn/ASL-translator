import matplotlib
matplotlib.use("Agg")

"""
python sign_train.py --dataset "Gesture Image Pre-Processed Data"/ --model output/activity3.model --label-bin output/lb3.pickle --epochs 2
"""

# import the necessary packages
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras import losses
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#LABELS = set(["0","-","1","2","3"])
print("Loading images")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
count = 0
for i in imagePaths:
	p =i.split(os.path.sep)[1]
	q = p.split(".")[0]
	if int(q) >50:
		continue
	a = i.split(os.path.sep)[0]
	b = a.split('/')
	label = b[-1]
	image = cv2.imread(i)
	image = cv2.resize(image, (224,224))
	grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(T,image) = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
	data.append(image)
	labels.append(label)
	count = count+1

data = np.array(data)
print(data.shape)
labels = np.array(labels)
print(labels.shape)

lb = LabelBinarizer()
labels = lb.fit_transform(labels) # One hot encoding of LABELS

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=10)

trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
#mean = np.array([123.68, 116.779, 103.939], dtype="float32")
#trainAug.mean = mean
#valAug.mean = mean

# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224,1)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()


#model = keras.Sequential([keras.layers.Flatten(input_shape = (224,224)), #keras.layers.Dense(128,activation=tf.nn.relu), keras.layers.Dense(37,activation = #tf.nn.softmax)])
#
#print("[INFO] compiling model...")
#opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
#model.compile(loss="categorical_crossentropy", optimizer=opt,
#	metrics=["accuracy"])



# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
#print("[INFO] training head...")
#H = model.fit(trainX, trainY,epochs=args["epochs"])
#
#print("[INFO] evaluating network...")
#predictions = model.predict(testX)
#print(classification_report(testY.argmax(axis=1),
#	predictions.argmax(axis=1), target_names=lb.classes_))
#

# serialize the model to disk
#print("[INFO] serializing network...")
#model.save(args["model"])
#
## serialize the label binarizer to disk
#f = open(args["label_bin"], "wb")
#f.write(pickle.dumps(lb))
#f.close()
#

import numpy as np
import random
import csv
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def simple_model():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))
	return model

def le_net(cropping = False):
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	if cropping:
		model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

def nvidia(cropping=True):
	keep_prob = 1.0
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	if cropping:
		model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Convolution2D(24//2,kernel_size=(5,5),activation="relu",padding='valid'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(36//2,kernel_size=(5,5),activation="relu",padding='valid'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(48//2,kernel_size=(3,3),activation="relu",padding='valid'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(64//2,kernel_size=(3,3),activation="relu",padding='valid'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(64//2,kernel_size=(1,1),activation="relu",padding='valid'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(keep_prob))
	model.add(Dense(50))
	model.add(Dropout(keep_prob))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

#model = le_net(False)
model = nvidia()
model.summary()

def should_use_measurement():
	return random.randint(0, 2) < 1 # use a third of measurements

#data_dir = "/home/mw/data"
#data_dir = "/home/mw/training2"

steering_correction = 1.3
data_dir_prefix = "/home/mw/p3"
data_dirs = ["data",
			 "fix1",
			 "fix3",
			 #"recovery",
			]

images = []
measurements = []
previous_length = 0

from keras.preprocessing.image import img_to_array, load_img
for current_dir in data_dirs:
	data_dir = os.path.join(data_dir_prefix, current_dir)
	t1 = time.time()
	lines = []
	with open(os.path.join(data_dir, "driving_log.csv")) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	random.seed(42)
	skipped = 0
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		#image_left = cv2.imread(os.path.join(data_dir, "IMG", line[1].split('/')[-1]))
		#image_right = cv2.imread(os.path.join(data_dir, "IMG", line[2].split('/')[-1]))
		current_path = os.path.join(data_dir, "IMG", filename)
		#image = cv2.imread(current_path)
		image = load_img(current_path)
		image = img_to_array(image)
		#print(image)
		measurement = float(line[3])
		measurement_left = measurement * steering_correction
		measurement_right = -measurement * steering_correction
		if measurement != 0.0 or should_use_measurement():
			measurements.append(measurement)
			images.append(image)
			# add flipped images
			#images.append(cv2.flip(image,1))
			images.append(np.fliplr(image))
			measurements.append(measurement*-1.0)
			# add left/right images
		#	measurements.append(measurement_left)
		#	images.append(image_left)
		#	measurements.append(measurement_right)
		#	images.append(image_right)
		#	measurements.append(-1.0*measurement_left)
		#	images.append(cv2.flip(image_left,1))
		#	measurements.append(-1.0*measurement_right)
		#	images.append(cv2.flip(image_right,1))
		else:
			skipped += 1

	assert len(images) == len(measurements)
	print("Imported {} samples in {:.2f}s from {}.".format(len(images) - previous_length, time.time()-t1, data_dir))
	previous_length = len(images)
	if skipped > 0:
		print("Skipped {} images with steering of zero.".format(skipped))
print("\n\n")

X_train = np.array(images)
y_train = np.array(measurements)


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')
exit()

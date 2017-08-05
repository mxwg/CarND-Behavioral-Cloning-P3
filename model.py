import numpy as np
import csv
import cv2
import os
import time

data_dir = "/home/mw/driving_data"

t1 = time.time()
lines = []
with open(os.path.join(data_dir, "driving_log.csv")) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = os.path.join(data_dir, "IMG", filename)
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	# add flipped images
	images.append(cv2.flip(image,1))
	measurements.append(measurement*-1.0)

assert len(images) == len(measurements)
print("Imported {} samples in {:.2f}s.".format(len(images), time.time()-t1))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def simple_model():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))
	return model

def le_net():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

model = le_net()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()

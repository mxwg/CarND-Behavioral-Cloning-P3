import numpy as np
import random
import csv
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D

create_images = True

def nvidia(cropping=True):
	"""Create a Nvidia convolutional network with reduced number of convolutional layers."""
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
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

model = nvidia()
model.summary()
if create_images:
	from keras.utils import plot_model
	plot_model(model, to_file='model.png', show_shapes=True)
	print("Wrote model.png.")

def use_measurement_anyway():
	return random.randint(0, 2) < 1 # use a third of measurements

data_dir_prefix = "/home/mw/p3"
data_dirs = ["data",
			 "fix1",
			 "fix2",
			 "fix3",
			]

images = []
measurements = []
previous_length = 0

# Load images and measurements from all directories in data_dirs
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
		current_path = os.path.join(data_dir, "IMG", filename)
		image = load_img(current_path)
		image = img_to_array(image)
		measurement = float(line[3])
		if measurement != 0.0 or use_measurement_anyway(): # don't use all of the zero steering data
			measurements.append(measurement)
			images.append(image)
			# add flipped images
			images.append(np.fliplr(image))
			measurements.append(measurement*-1.0)
		else:
			skipped += 1

	assert len(images) == len(measurements)
	print("Imported {} samples in {:.2f}s from {}.".format(len(images) - previous_length, time.time()-t1, data_dir))
	previous_length = len(images)
	if skipped > 0:
		print("Skipped {} images with steering of zero.".format(skipped))
print("\n\n")

if create_images:
	n = 1520
	# Save example images
	example = images[n]
	example_cropped = example[50:160-20,:]
	import scipy.misc
	scipy.misc.imsave('example_image.png', example)
	scipy.misc.imsave('example_image_cropped.png', example_cropped)
	print("Wrote example images.")

# Set up the training data
X_train = np.array(images)
y_train = np.array(measurements)


# Train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

# Save the model
model.save('model.h5')

exit()

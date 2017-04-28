import csv
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import pickle

# Preprocessing the image
def preprocessimage(img_filename):

	img = cv2.imread(img_filename)
	img2 = img[66:132, :, :]
	res = cv2.resize(img2,(200, 66), interpolation = cv2.INTER_CUBIC)
	# res = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
	res = res / 255 - 0.5
	return np.array(res)

def get_data(folder, drivinglogfilename, split_ratio):
	
	lines = []
	with open (folder + drivinglogfilename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	print('CSV File uploaded!')

	center = []
	left = []
	right = []
	measurements = []

	for line in lines:
		center_filename = folder + line[0]
		left_filename = folder + line[1]
		right_filename = folder + line[2]

		center.append(center_filename)
		left.append(left_filename)
		right.append(right_filename)

		measurement = float(line[3])
		measurements.append(measurement)

	# Concatinating the image names
	X = center + left + right

	# Concatinating the theta values into y_train
	y = np.array(measurements + [x+0.25 for x in measurements] + [x-0.25 for x in measurements])

	# print('Uploaded data...')
	# print('Size of training set is', y.shape[0])

	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = split_ratio, random_state = 69)

	return X_train, X_valid, y_train, y_valid

def image_generator(data, labels, batch_size):
	# The input to this should be the file names of the images 
	# to be read and the values of the labels i.e. theta
    while 1:
    	# Reshuffle data and labels
    	data, labels = shuffle(data, labels, random_state=69)

    	# Define placeholders
    	minidata = np.empty([batch_size, 66, 200, 3])
    	minilabels = np.empty([batch_size, 1])

    	# Loop over and assign the minibatch
    	num_batches = int(np.floor(labels.shape[0]/batch_size))
    	for i in range(num_batches):
    		for j in range(batch_size):
    			minidata[j] = preprocessimage(data[i*batch_size + j])

    			minilabels[j] = labels[i*batch_size + j]
    		yield minidata, minilabels

def get_validation_images(filenames):
	images = np.empty([len(filenames), 66, 200, 3])
	i = 0
	for file in filenames:
		images[i] = preprocessimage(file)
	return np.array(images)

def nvidia_model(summary = True):
	# Nvidia architecture
	model = Sequential()

	model.add(Conv2D(24, (5, 5), padding="same", strides=(2,2), activation="elu", input_shape=(66, 200, 3)))
	model.add(SpatialDropout2D(0.2))
	model.add(Conv2D(36, (5, 5), padding="same", strides=(2,2), activation="elu"))
	model.add(SpatialDropout2D(0.2))
	model.add(Conv2D(48, (5, 5), padding="valid", strides=(2,2), activation="elu"))
	model.add(SpatialDropout2D(0.2))
	model.add(Conv2D(64, (3, 3), padding="valid", activation="elu"))
	model.add(SpatialDropout2D(0.2))
	model.add(Conv2D(64, (3, 3), padding="valid", activation="elu"))
	model.add(SpatialDropout2D(0.2))

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(100, activation="elu"))
	model.add(Dense(50, activation="elu"))
	model.add(Dense(10, activation="elu"))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	if(summary):
		model.summary()

	model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.001), metrics=['mae'])

	return model
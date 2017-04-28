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
from random import randint
import pickle

# Preprocess image to visualize data
def preprocessimagevisualization(img_filename):

	img = cv2.imread(img_filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img2 = img[66:132, :, :]
	res = cv2.resize(img2,(200, 66), interpolation = cv2.INTER_CUBIC)
	# res = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
	# res = res / 255 - 0.5
	return res

# Preprocessing the image
def preprocessimage(img_filename):

	img = cv2.imread(img_filename)
	img2 = img[66:132, :, :]
	res = cv2.resize(img2,(200, 66), interpolation = cv2.INTER_CUBIC)
	# res = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
	res = res / 255 - 0.5
	return np.array(res)

def get_data(folder, drivinglogfilename, split_ratio, low_steer, drop_ratio, visualize = True):
	
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

	index = randint(0, len(measurements)-1)
	plt.figure()
	plt.subplot(131)
	plt.imshow(preprocessimagevisualization(left[index]))
	plt.title('Theta = ' + str(measurements[index]+0.25))
	plt.subplot(132)
	plt.imshow(preprocessimagevisualization(center[index]))
	plt.title('Theta = ' + str(measurements[index]))
	plt.subplot(133)
	plt.imshow(preprocessimagevisualization(right[index]))
	plt.title('Theta = ' + str(measurements[index]-0.25))

	plt.figure()
	plt.subplot(121)
	plt.imshow(plt.imread(center[index]))
	plt.subplot(122)
	plt.imshow(preprocessimagevisualization(center[index]))

	flipped_img = cv2.flip(preprocessimagevisualization(left[index]), 1)
	plt.figure()
	plt.subplot(121)
	plt.imshow(preprocessimagevisualization(left[index]))
	plt.title('Theta = ' + str(measurements[index]+0.25))
	plt.subplot(122)
	plt.imshow(flipped_img)
	plt.title('Flipped Theta = ' + str(-(measurements[index]+0.25)))

	num_bins = 100
	avg_samples_per_bin = len(measurements)/num_bins
	hist, bins = np.histogram(np.abs(np.array(measurements)), num_bins)
	width = 0.7 * (bins[1] - bins[0])
	centerbin = (bins[:-1] + bins[1:]) / 2

	plt.figure()
	plt.bar(centerbin, hist, align='center', width=width)
	plt.title('Before')

	# Drop low steering angles
	center, left, right, measurements = shuffle(center, left, right, measurements)
	prev_len = len(center)
	# Now drop first 70% of low steering angles
	total_num_of_low_angle_steers = len([value for value in measurements if abs(value) < low_steer])
	print('Total Data =', prev_len)
	print('Total number of low steering angles =', total_num_of_low_angle_steers)
	num_to_be_dropped = int(drop_ratio*total_num_of_low_angle_steers)
	print('Number to be dropped =', num_to_be_dropped)
	count = 0
	indices_to_be_deleted = []
	for i in range(len(measurements)):
		if (abs(measurements[i]) < low_steer):
			count = count + 1
			if (count > num_to_be_dropped):
				break
			indices_to_be_deleted.append(i)
			#del center[i], left[i], right[i], measurements[i]
	# flip list to delete last element first
	indices_to_be_deleted = indices_to_be_deleted[::-1]

	# delete the low steering angles
	for i in indices_to_be_deleted:
		del center[i], left[i], right[i], measurements[i]

	print('length before =', prev_len, ',length after =', len(center))

	num_bins = 100
	avg_samples_per_bin = len(measurements)/num_bins
	hist, bins = np.histogram(np.abs(np.array(measurements)), num_bins)
	width = 0.7 * (bins[1] - bins[0])
	centerbin = (bins[:-1] + bins[1:]) / 2

	plt.figure()
	plt.bar(centerbin, hist, align='center', width=width)
	plt.title('After')

	# Concatinating the image names
	X = center + left + right

	# Concatinating the theta values into y_train
	y = np.array(measurements + [x+0.25 for x in measurements] + [x-0.25 for x in measurements])

	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = split_ratio) #, random_state = 69)

	if(visualize):
		plt.show()

	return X_train, X_valid, y_train, y_valid

def image_generator(data, labels, batch_size):
	# The input to this should be the file names of the images 
	# to be read and the values of the labels i.e. theta
    while 1:
    	# Reshuffle data and labels
    	data, labels = shuffle(data, labels) #, random_state=69)

    	# Define placeholders
    	minidata = np.empty([batch_size, 66, 200, 3])
    	minilabels = np.empty([batch_size, 1])

    	# Loop over and assign the minibatch
    	num_batches = int(np.floor(labels.shape[0]/batch_size))
    	for i in range(num_batches):
    		for j in range(batch_size):
    			# flip 30% of the times
    			if(np.random.uniform() < 0.3):
    				minidata[j] = cv2.flip(preprocessimage(data[i*batch_size + j]), 1)
	    			minilabels[j] = -labels[i*batch_size + j]    			
    			else:
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
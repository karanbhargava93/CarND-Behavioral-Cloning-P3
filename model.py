from utils import *

# Get data and labels into training and validation sets with 20% split
X_train, X_valid, y_train, y_valid = get_data('./data_udacity/data1/', 'driving_log.csv', 
												split_ratio = 0.2, 
												low_steer = 0.01, 
												drop_ratio = 0.95,
												visualize = False) 
# ('', './data/driving_log.csv', 0.2)#('./data_udacity/data1/', 'driving_log.csv', 0.2)

# Instantiate model
model = nvidia_model(summary = False)

# Parameters
BATCH_SIZE = 32
EPOCHS = 50
num_per_epoch = int(y_train.shape[0]/BATCH_SIZE)
validation_images = get_validation_images(X_valid)

# Train the model
model.fit_generator(
	image_generator(X_train, y_train, BATCH_SIZE), 
	steps_per_epoch=num_per_epoch, 
	epochs=EPOCHS, 
	validation_data=(validation_images, y_valid), 
	validation_steps=len(X_valid)
	)

print('Start Save...')
model.save('model.h5')
print('Saved Network')
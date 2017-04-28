# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/arch.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 (a video recording of my vehicle driving autonomously around the track for one full lap)
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've used the [End to End Learning Architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidia.

My model consists of a convolution neural network with 5x5 filter sizes and depths 24, 36 and 48 (`utils.py` lines 124 to 128). I've used 3x3 filter size with depth of 64 twice after the previous filters. Every layer has been passed into a dropout layer so as to avoid overfitting. Then I flattened out the data and used dense layers with ELU activations to get the fianl steering angle.

The model includes Exponential Linear Units (ELU) layers to introduce nonlinearities (in `model.py` lines 124 to 139), and the normalized data is fed to model using the preprocessed images from the function `preprocessimage` (in `utils.py` lines 27 to 34). It also uses dropout layers after every almost every layer to avoid overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`utils.py` lines 125 to 140). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (`utils.py` line 90). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually however the default learning rate (0.01) wasn't giving satisfactory results so I changed it to 0.001 (`utils.py` line 146).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

`
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 33, 100, 24)       1824      
_________________________________________________________________
spatial_dropout2d_1 (Spatial (None, 33, 100, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 50, 36)        21636     
_________________________________________________________________
spatial_dropout2d_2 (Spatial (None, 17, 50, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 23, 48)         43248     
_________________________________________________________________
spatial_dropout2d_3 (Spatial (None, 7, 23, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 21, 64)         27712     
_________________________________________________________________
spatial_dropout2d_4 (Spatial (None, 5, 21, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 19, 64)         36928     
_________________________________________________________________
spatial_dropout2d_5 (Spatial (None, 3, 19, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3648)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 3648)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               364900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_2 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 501,819
Trainable params: 501,819
Non-trainable params: 0
_________________________________________________________________
`

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

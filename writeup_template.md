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
[theta_dist_before]: ./examples/theta_dist_before.png "theta_dist_before"
[theta_dist_after]: ./examples/theta_dist_after.png "theta_dist_after"
[crop]: ./examples/crop.png "Cropped Image"
[flip]: ./examples/flip.png "Flipped Image"
[sample_theta]: ./examples/sample_theta.png "Left Center Right"
[video]: video.mp4 "video"

[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Video Output

A video of a successful lap around track one is given as [video] from running the scripts given by Udacity.

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

I've used the [End to End Learning Architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidia. The complete model is given as a function named `nvidia_model` in `utils.py` lines 181 through 209

My model consists of a convolution neural network with 5x5 filter sizes and depths 24, 36 and 48 (`utils.py` lines 185 to 189). I've used 3x3 filter size with depth of 64 twice after the previous filters. Every layer has been passed into a dropout layer so as to avoid overfitting. Then I flattened out the data and used dense layers with ELU activations to get the final steering angle.

The model includes Exponential Linear Units (ELU) layers to introduce nonlinearities (in `model.py` lines 185 to 200), and the normalized data is fed to model using the preprocessed images from the function `preprocessimage` (in `utils.py` lines 27 to 34). It also uses dropout layers after every almost every layer to avoid overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`utils.py` lines 185 to 202). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (`utils.py` line 143). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually however the default learning rate (0.01) wasn't giving satisfactory results so I changed it to 0.001 (`utils.py` line 207). The model was run for 50 epochs keeping in mind the validation mae as the measure of overfitting and underfitting.

#### 4. Appropriate training data

I used the dataset given by udacity in conjuction with various techniques which are discussed below to yield a decent dataset which can be used to create a working model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet architecture to get a feel of how keras would work. Obviously it didn't work well. Then I searched for previously built networks for autonomous cars and came across the paper by Nvidia. They had used the same data as us and gotten neat results. So I thought that its a nice place to start.

To combat the overfitting, I modified the model so that it used dropout layers after almost all layers. This made it very unlikely for the model to overfit.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. This was done in the `get_data` function in `utils.py` line 143. The network was trained till the mean absolute error in the validation set started ramping up (EPOCH 50). 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and to improve the driving behavior in these cases, I modified the image generator to augment the dataset with the techniques given below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`utils.py` lines 181 through 209) consisted of the nvidia architecture with the following layers and layer sizes given below: 

```
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
```

Here is a visualization of the architecture from the Nvidia paper.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I have used the dataset given by udacity to train my model. However the model wasn't trained well using the data as is. I had to augment, crop, resize and drop low steering angles so as to arrive at a good dataset for the model to work on.

#### A. Dropping Low Angles

![alt text][theta_dist_before]

The udacity dataset had the above distribution for the frequency of images versus the absolute theta angle. As we can clearly see, most of the data is concentrated at very small angles. We need to remove most of these low steering angles so as to let our model generalize better and not be influenced much from the low angles only. Otherwise our model would be very good at driving straight but will perform catastrophically at tight curves. The code used to randomly drop these angles from the dataset is given in the `get_data` function in the `utils.py` at lines 100 to 125. The distribution after dropping the angles is given below.

![alt text][theta_dist_after]

#### B. Cropping Images

The model was easily distracted by the trees and scenery which was above the road. So I decided to crop the image and resize it to 66 x 200 which was the input used in the Nvidia paper. This is a sample of the original and cropped image. The code for this is in the `preprocessimage` function in `utils.py` at lines 27 to 34.

![alt text][crop]

#### C. Flipping Images

Its very intuitive that if the control action was theta for an image then for the vertically flipped image the control signal would be -theta. I've flipped images randomly 30% of the time in the `image_generator` function in `utils.py` from line 166 to 171.

![alt text][flip]

#### D. Using Left and Right Camera Images
The left and right camera images can be treated as the center camera image displaced by a certain distance. The steering angle with both the right and the left camera is offset by a certain quantity from the steering angle of the center camera if we view them independantly to increase our datasize by 200%. This quantity was determined emperically and was close to 0.25. This has been done in the `get_data` function in `utils.py` from line 137 to 141.

![alt text][sample_theta]

#### E. Image Generator

The model has been trained with an image generator which takes the list of filenames and the batch size to yield a minibatch of data. It reduces the memory used and hence can be used with most computers when the batchsize is changed. If we do not use this then we might run into situations with memoryerrors. The code for the `image_generator` function is in `utils.py` from line 150 to 172.

#### F. Miscellaneous

I finally randomly shuffled the dataset and put 20% of the data into a validation set in the function `get_data` in `utils.py` from line 143.

I used this training data for training the model through the `image_generator`. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by the mae on the validation  I used an adam optimizer with 0.001 as the initial learning rate so that manually training the learning rate wasn't necessary.

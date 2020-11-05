# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image_data/nVIDIA_model.PNG "Model Visualization"
[image2]: ./image_data/center_2020_10_13_09_50_01_163.jpg "curve driving"
[image3]: ./image_data/center_2020_10_13_09_49_09_149.jpg "center camera image"
[image4]: ./image_data/left_2020_10_13_09_49_09_149.jpg "left camera image"
[image5]: ./image_data/right_2020_10_13_09_49_09_149.jpg "right camera image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_shoito.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network from nVIDIA CNN (model.py lines 112-124)
Because LeNet model failed automonous driving , I Selected model below.
![alt text][image1]

The data is normalized in the model using a Keras lambda layer (code line 109). 

### 2. Attempts to reduce overfitting in the model

The model contains regularizer(l2) in order to reduce overfitting (model.py lines 112,114,116,118,119,121,122,123). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 127). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 126).

When the epoch was 5 or more, overfitting started, so the epoch was set to 5 times.

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used sample
driving data, because there are a lot of beneficial driving data.
In order to increase data , I inverted the above data.(model.py lines 49-53)

And I used curve driving data , because I was worried about curve driving.

![alt text][image2]

I used 3 camera data(front/left/right) about above driving data.(model.py lines 39-41 , 82-84)

center camera image

![alt text][image3] 

left camera image

![alt text][image4] 

right camera image

![alt text][image5]



import csv
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

# Get cvs file data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)

# Get image data        
car_images = []
steering_angles = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path_center = './data/IMG/' + filename

    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path_left = './data/IMG/' + filename

    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path_right = './data/IMG/' + filename
    
    steering_center = float(line[3])    
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # read in images from center, left and right cameras
    img_center = np.asarray(Image.open(current_path_center))
    img_left =  np.asarray(Image.open(current_path_left))
    img_right = np.asarray(Image.open(current_path_right))
    
    # add images and angles to data set
    car_images.extend([img_center, img_left, img_right])
    steering_angles.extend([steering_center, steering_left, steering_right])

# Data Augmentation
augmented_images, augmented_measurements = [],[]
for image, measurement in zip(car_images, steering_angles):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)

    
# My Driving Data Load
lines = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path_center = './my_data/IMG/' + filename

    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path_left = './my_data/IMG/' + filename

    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path_right = './my_data/IMG/' + filename
    
    steering_center = float(line[3])    
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # read in images from center, left and right cameras
    img_center = np.asarray(Image.open(current_path_center))
    img_left =  np.asarray(Image.open(current_path_left))
    img_right = np.asarray(Image.open(current_path_right))
    
    # add images and angles to data set
    augmented_images.extend([img_center, img_left, img_right])
    augmented_measurements.extend([steering_center, steering_left, steering_right])
    
#register data 
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Model Part
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.regularizers import l2

# Create the Sequential model
model = Sequential()

# image Cropping
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# image Normalization
model.add(Lambda(lambda x: (x / 127.5) - 1.0))

# create nNIDIA model
model.add(Convolution2D(24,5,5, border_mode='valid', W_regularizer=l2(0.001), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(36,5,5, border_mode='valid', W_regularizer=l2(0.001), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(48,5,5, border_mode='valid', W_regularizer=l2(0.001), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001), activation='relu'))
model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001), activation='relu'))
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(0.001), activation='relu'))
model.add(Dense(50, W_regularizer=l2(0.001), activation='relu'))
model.add(Dense(10, W_regularizer=l2(0.001), activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
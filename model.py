
##### Start by importing all the required modules
import seaborn as sns
import pandas as pd
import os
import ntpath
import csv
import cv2
import keras
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from imgaug import augmenters as iaa
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

##### Laod the data
datadir = os.path.join('/home', 'workspace', 'CarND-Behavioral-Cloning-P3','data')
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)

### Extracting center, left and right columns and creat a dataframe
data['center'] = data['center'].apply(lambda x:ntpath.split(x)[1])
data['left'] = data['left'].apply(lambda x:ntpath.split(x)[1])
data['right'] = data['right'].apply(lambda x:ntpath.split(x)[1])
image_df = data[['center','left','right']]

### adding steering angle correction to the left and right and creating steering dataframe
angle_correction = 0.25
data['steering_left'] = data['steering'] + angle_correction
data['steering_right'] = data['steering'] - angle_correction
steering_df = data[['steering','steering_left','steering_right']]

###### Put together image file paths and corresponding steering data as one data frame
clean_data = pd.concat([image_df, steering_df], axis=1, sort=False)


def load_img_steering_filtered(imgdir, data):
    """ creat image path files and steering angles as an array """
  image_path = []
  steering = []
  for i in range(len(data)):
    row_data = data.iloc[i]
    center, left, right = row_data[0], row_data[1], row_data[2]
    ###Center image data
    angle = float(row_data[3])
    if abs(angle) < 0.1 :      # skip angles ~0
      continue
    image_path.append(os.path.join(imgdir, center.strip()))
    steering.append(angle)
    # left image data
    image_path.append(os.path.join(imgdir, left.strip()))
    steering.append(float(row_data[4]))
    #right image data 
    image_path.append(os.path.join(imgdir, right.strip()))
    steering.append(float(row_data[5]))
  image_paths_f = np.array(image_path)
  steerings_f = np.array(steering)
  return image_paths_f, steerings_f
image_paths_f, steerings_f = load_img_steering_filtered(datadir + '/IMG', clean_data)


### Defining the desired input image shape to the model
image_H, image_W, image_CH = 66, 200, 3  #### Per Nvidia model
INPUT_SHAPE = (image_H, image_W, image_CH)



def process_image(image):
    """ Pre-process the image by cropping, resizing and
        changing the color map from RGB to YUV
    """
    image = image[60:135, :, :]
    image = cv2.resize(image, (image_W, image_H), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


### Augmenting images
def zoom(image):
    """ augmenting the image by zooming
        the input image
    """
  if np.random.rand() < 0.5:
    zoom = iaa.Affine(scale=(1, 1.5))
    image = zoom.augment_image(image)
  return image

def random_brightness(image):
  """ Randomly alters the brightness of the input image
  """
  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  ratio = 1.0 + (np.random.rand() - 0.5)
  hsv[:,:,2] =  hsv[:,:,2] * ratio
  image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  return image


def flip_horz(image, steering_angle):
  """
    Randomly flipping the input image horizontaly and adjusts the steering
  """
  if np.random.rand() < 0.5:
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
  return image, steering_angle


def augment(image, steering_angle, range_x=100, range_y=10):
    """
    Randomly augments an image
    """
    image = mpimg.imread(image)
    image = zoom(image)
    image = random_brightness(image)
    image, steering_angle = flip_horz(image, steering_angle)
    return image, steering_angle


### Defining parameters
batch_size = 100
samples_per_epoch = 500
nb_epoch = 5

def batch_generator(image_paths, steerings, batch_size, istraining):
    """
    Generate a training image given image paths and the associated steering angles
    """
    images = np.empty([batch_size, image_H, image_W, image_CH])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            if istraining and np.random.rand() < 0.6:
                image, steering_angle = augment(image_paths[index], steerings[index])
            else:
              image = mpimg.imread(image_paths[index])
              steering_angle = steerings[index] 
            images[i] = process_image(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

        
### Split the data to train and validation data         
X_train, X_valid, y_train, y_valid = train_test_split(image_paths_f,steerings_f,test_size=0.2, random_state=40)


def nvidia_model():
    """ NVIDIA model used for training """
  model = Sequential()
  model.add(Lambda(lambda x: (x / 127.5)-1 , input_shape=INPUT_SHAPE))
  model.add(Conv2D(24, (5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(64, (3,3), activation='elu'))
  model.add(Conv2D(64, (3,3), activation='elu'))

  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dense(1))
  optimizer = keras.optimizers.Adam(lr=0.0001)
  model.compile(loss='mse', optimizer=optimizer)
  return model

### defining model checkpoint to save model and compiling
def train_model(model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model4-{val_loss:03f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    
    model.fit_generator(batch_generator(X_train, y_train, batch_size, True),
                        samples_per_epoch,
                        nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(X_valid, y_valid, batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

### Perfom model training
model = nvidia_model()
train_model(model, X_train, X_valid, y_train, y_valid)








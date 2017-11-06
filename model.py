import tensorflow as tf
import keras
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.utils
import sklearn.model_selection
import cv2
print('Tensorflow version {0}'.format(tf.__version__))
print('Keras version {0}'.format(keras.__version__))

def change_brightness(img, lower_bound=0, upper_bound=255):
    '''
    Uniformly changes the brightness of an image randomly without overflowing.
    Change in brightness limited by original brightness of image
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2]/2 + np.random.randint(lower_bound, upper_bound)/2
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def generator(file_paths, samples, batch_size, lr_shift):
    num_samples = len(samples)
    if len(samples) != len(file_paths):
        print('TODO: Error Message, incompatible arrays {0} {1}'.format(len(samples), len(file_paths)))
    while True:
        file_paths, samples = sklearn.utils.shuffle(file_paths, samples)
        for offset in range(0, num_samples, batch_size):
            batch_files = file_paths[offset:offset + batch_size]
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            measurements = []
            for batch_file in batch_files:
                #Read images from file
                center_file  = batch_file[0]
                left_file    = batch_file[1]
                right_file   = batch_file[2]
                center_image = mpimg.imread(center_file)
                left_image   = mpimg.imread(left_file)
                right_image  = mpimg.imread(right_file)

                #Flip images
                flip_center  = np.fliplr(center_image)
                flip_left    = np.fliplr(left_image)
                flip_right   = np.fliplr(right_image)

                #Change brightness of images
                center_image = change_brightness(center_image)
                left_image   = change_brightness(left_image)
                right_image  = change_brightness(right_image)
                flip_center  = change_brightness(flip_center)
                flip_left    = change_brightness(flip_left)
                flip_right   = change_brightness(flip_right)
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                images.append(flip_center)
                images.append(flip_left)
                images.append(flip_right)
            
            for batch_sample in batch_samples:
                steer_angle  = batch_sample[0]

                #Steering angle offsets
                measurements.append(steer_angle)
                measurements.append(steer_angle+lr_shift)
                measurements.append(steer_angle-lr_shift)
                measurements.append(-steer_angle)
                measurements.append(-steer_angle-lr_shift)
                measurements.append(-steer_angle+lr_shift)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# Separate CSV into numpy array of strings and numpy array of floats
file_paths = np.genfromtxt('./driving_data/driving_log.csv', dtype=np.str, usecols=(0,1,2), delimiter=',')
homogenous_samples = np.genfromtxt('./driving_data/driving_log.csv', delimiter=',', usecols=(3,4,5,6))
# Compute sample weights
steering_angles = homogenous_samples[:,0]

BATCH_SIZE = 32
SHIFT = 0.1
train_files, val_files, train_samples, val_samples = \
    sklearn.model_selection.train_test_split(file_paths, homogenous_samples, test_size=0.2)
train_generator = generator(train_files, train_samples, batch_size=BATCH_SIZE, lr_shift=SHIFT)
validation_generator = generator(val_files, val_samples, batch_size=BATCH_SIZE, lr_shift=SHIFT)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import MaxPooling2D, Cropping2D, Dropout
from keras.utils import plot_model

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24, (5,5), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(36, (5,5), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(48, (5,5), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='elu'))
model.add(Dense(64, activation='elu'))
model.add(Dense(32, activation='elu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=0.0001))
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=len(train_samples)/BATCH_SIZE,
                                     validation_data=validation_generator, 
                                     validation_steps=len(val_samples)/BATCH_SIZE,
                                     epochs=5, 
                                     verbose=2)
model.save('model{0}.h5'.format(np.datetime_as_string(np.datetime64('now'))))
plot_model(model, to_file='model{0}.png'.format(np.datetime_as_string(np.datetime64('now'))))

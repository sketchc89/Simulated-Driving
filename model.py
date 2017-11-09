import tensorflow as tf
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import MaxPooling2D, Cropping2D, Dropout
from keras.utils import plot_model
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
        print('ERROR: Incompatible arrays {0} {1}'.format(len(samples), 
                                                          len(file_paths)))
        return None
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
csv_fp = './driving_data/driving_log.csv'
file_paths = np.genfromtxt(csv_fp, dtype=np.str, usecols=(0,1,2), delimiter=',')
homogenous_samples = np.genfromtxt(csv_fp, delimiter=',', usecols=(3,4,5,6))

#Steering wheels stored in y
y = homogenous_samples[:,0]

#Separate into n_bins bins and attempt to get <= max_samples in each
n_bins = 50
y_bins = np.floor((y - np.min(y))*n_bins/(np.max(y) - np.min(y)))
max_samples = 500
pct_keep = []
for bin_num in range(n_bins+1):
    bin_n = y_bins == bin_num
    bin_count = bin_n.sum()
    pct_keep.append(min(max_samples/bin_count, 1)) 

#Generate random number for each sample, compare to percent.
#e.g. Bin of 5000 samples with 250 max will be 5%. Any number less than 5% will
#be saved all others will be deleted. Random numbers have equal distribution
#so will delete about 95% of samples. Indexes used for number of row to delete.
random_keep = np.random.rand(homogenous_samples.shape[0],)
class_dict = dict(zip(np.unique(y_bins), pct_keep))
keep_array = np.vectorize(class_dict.get)(y_bins)
delete_sample = random_keep > keep_array
indexes = np.arange(homogenous_samples.shape[0])
balanced_array = np.delete(homogenous_samples, indexes[delete_sample], axis=0)
balanced_files = np.delete(file_paths, indexes[delete_sample], axis=0)
print('Unbalanced samples: {0}'.format(len(homogenous_samples)))
print('Balanced samples: {0}'.format(len(balanced_array)))
BATCH_SIZE = 64
SHIFT = 0.25
EPOCHS = 10
train_files, val_files, train_samples, val_samples = \
    sklearn.model_selection.train_test_split(file_paths, homogenous_samples, 
                                             test_size=0.2)
train_generator = generator(train_files, train_samples, 
                            batch_size=BATCH_SIZE, lr_shift=SHIFT)
validation_generator = generator(val_files, val_samples, 
                                 batch_size=BATCH_SIZE, lr_shift=SHIFT)


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
                             epochs=EPOCHS, 
                             verbose=2)
time_str = np.datetime_as_string(np.datetime64('now'))
model.save('model{0}.h5'.format(time_str))
plot_model(model, to_file='model{0}.png'.format(time_str))
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.savefig('loss{}.png'.format(time_str))
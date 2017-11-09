#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./images/nvidia.png "End-end architecture from NVIDIA paper"
[recover_0]: ./images/resize_recovery_0.png "Recovery Image"
[recover_1]: ./images/resize_recovery_1.png "Recovery Image"
[recover_2]: ./images/resize_recovery_2.png "Recovery Image"
[center_1]: ./images/resize_center_1.png "Lane Centered"
[center_2]: ./images/resize_center_2.png "Lane Centered"
[architecture]: ./images/resize_model.png "Model Architecture"
[unbalanced]: ./images/unbalanced_large.png "Unbalanced dataset"
[balanced]: ./images/balanced_large.png "Balanced dataset"
[train_val_loss]: ./images/loss.png "Training and validation loss"
[preprocess]: ./images/flip_bright.png "Images cropped, flipped, and brightened/darkened"
[run3]: ./images/run3.gif "Run 3 gif"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The Behavioral_Cloning.ipynb is a more readable version of model.py.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA architecture](https://arxiv.org/abs/1604.07316). 

---

![NVIDIA Architecture][nvidia]

---

![Architecture][architecture]

---

The data is normalized with a lambda layer. The model uses convolution layers, ELU layers to introduce nonlinearity, and max pooling layers to reduce the size of the data and prevent overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains max pooling layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 121-8). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

![Architecture][run3]


#### 3. Model parameter tuning

The model used an Adam optimizer with a learning rate of 1e-4, (model.py line 130-54).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track in both directions, and driving similarly on the second track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it successfully implemented an end-end steering task similar to this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I tried adding dropout layers to reduce overfitting, but I the car didn't drive as well with them and the validation set did not show overfitting, so I left them out.

In order to balance my dataset I tried changing the sample weights so that turning was weighted more strongly than going straight. I divided the data into 10 bins, then used the relative number of samples in each bin to compute the appropriate sample weight. These sample weights were in the end not a great solution. The car tended to snake side-side when the weights of each bin were equalized. I believe the reason for this is because sharp turns are difficult to control in the simulator and the angles given to the simulator did not change smoothly.

I tried a second approach by setting a maximum number of samples to take from each bin. I chose to use 50 bins and 500 maximum samples per bin (25 and 250 for a smaller dataset). I calculated the percent of samples that would need to be deleted in order to bring the number of samples down to the maximum, then randomly deleted that percent of samples from each bin. The following show a histogram of unbalanced and balanced data.

![Unbalanced data][unbalanced]
![Balanced data][balanced]

Then I randomly changed the brightness of the images to make the model robust to changes in brightness.
The model was also trained forwards and backwards on the first track and the second track to try to reduce overfitting to the track.
The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 130-54) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a table of the architecture, the size of each layer, and the number of variables per layer



|Layer(type)|Output Shape|Param #|
|-|-|-|
|cropping2d_2(Cropping2D)|(None, 90, 320, 3)|0|
|lambda_2 (Lambda)|            (None, 90, 320, 3)|        0    |
|conv2d_6 (Conv2D)|(None, 90, 320, 24)|1824|
|max_pooling2d_6 (MaxPooling2 |(None, 45, 160, 24)|       0|
|conv2d_7 (Conv2D)            |(None, 45, 160, 36)|       21636|
|max_pooling2d_7 (MaxPooling2 |(None, 22, 80, 36) |       0|
|conv2d_8 (Conv2D)            |(None, 22, 80, 48) |       43248|
|max_pooling2d_8 (MaxPooling2 |(None, 11, 40, 48) |       0|
|conv2d_9 (Conv2D)            |(None, 11, 40, 64) |       27712|
|max_pooling2d_9 (MaxPooling2 |(None, 5, 20, 64)  |       0     |
|conv2d_10 (Conv2D)           |(None, 5, 20, 64)  |       36928  |
|max_pooling2d_10 (MaxPooling |(None, 2, 10, 64)  |       0       |
|flatten_2 (Flatten)          |(None, 1280)       |       0        |
|dense_5 (Dense)              |(None, 128)        |       163968|
|dense_6 (Dense)              |(None, 64)         |       8256|
|dense_7 (Dense)              |(None, 32)         |       2080|
|dense_8 (Dense)              |(None, 1)          |       33|
|Total params: 305,685|||
|Trainable params: 305,685|||
|Non-trainable params: 0|||


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving. I used my computer mouse to drive. The quality of driving wasn't exceptional, and its hard to tell how the final model is affected by this. People on the Udacity forums mentioned that using a controller and a higher frequency mode worked much better.

![Centered][center_1]

![Centered][center_2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the lane if it diverged from the center. These images show what a recovery looks like starting from the right side on the outside and inside of a turn:

![Outside turn][recover_1]

![Inside turn][recover_2]

Then I repeated this process on track one in the opposite direction (clockwise vs counter-clockwise) in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would prevent the car from being biased towards turning in one direction and because I have a bias towards driving on the right side of the road.

The cameras on the left and right side are not at the same position as the center. For a given steering angle they will be shifted laterally from the center. The camera is shifted approximately 50px laterally. A mid-size car is 1.8m wide. In order to readjust to the center in 1s at 20 mph, the car will travel 8.9m. The angle (from the arcsin) is 6.5 degrees which is approximately 0.25 in the unity car model. Steering angle measurements from the left and right side were adjusted by +0.25 for the left side and -0.25 for the right side.

After the collection process, I had about 10k data points. After subsampling described above, I had about 3k data points. That was doubled to 6k by flipping the images. It isn't exact since I probabilistically sub-sampled overrepresented angles. I then preprocessed the data by cropping the top and bottom off the image which were assumed to be unimportant to following the lines. The following set of images is an example of one time snapshot in time from the left, center, and right sides.

![Augmented data][preprocess]

I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the image below. I used an Adam optimizer with a learning rate of 1e-4 though the learning rate is adjusted by the algorithm from this point.

![Training-validation loss][train_val_loss]

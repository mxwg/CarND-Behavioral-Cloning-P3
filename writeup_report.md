# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted and Code Quality

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Note that this was developed on Ubuntu without the CarND environment, so it used more up-to-date version of tensorflow and Keras.
I still used the Keras layer naming as shown in the lectures, so I think this code should also run with the older Keras version.

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Model Architecture

I'm using a modified version of the Nvidia architecture shown in the lecture.

I've employed cropping of the input image and halved the depth of each of the convolutional layers (model.py lines 11 - 31).

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (model.py line 16). 

#### Attempts to reduce overfitting in the model

I've employed the `validation_split` functionality of Keras to split the data into training and validation sets (model.py line 93).

The depth of all convolutional layers of the Nvidia architecture was halved to avoid overfitting.

During training, I evaluated whether training loss and validation loss were similar and modified the architecture when they were too far apart.

#### Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

#### Training data

The training data is based on the data downloadable from Udacity and expanded with a few custom recordings for steep curves.

### Architecture and Training Documentation

#### Solution design approach

I followed the videos in first implementing a simple dense network, then the LeNet architecture and finally the Nvidia architecture.

The largest problem in this project was creating good-enough training data. I first started with recording my own driving using a keyboard.
A lot of experimentation with the architecture and data augmentation and creation techniques (left/right cameras, recovery driving) did not lead to satisfying results.

I then switched to using the Udacity data set. The main problem now was that the model did not steer aggressively enough in steep curves, thus driving off the road and into the lake.

This led me to record around 9 additional runs just driving through such steep curves.
I also used the mouse instead of the keyboard after reading about the keyboard problems on the forum.

These additional data helped the model learn the correct behavior for these curves.

The model is now able to drive the vehicle around the track without leaving the road (as shown in the two laps recorded for the video).
The speed of the vehicle was adapted from 9 to 25 in drive.py.

#### Final model architecture

The final model architecture is a modified Nvidia architecture as described above.

#### Creation of the training set and the training process

The additional curve runs were recorded in the following way:
I drove to before the beginning of the curve and stayed on the side of the road. 
During the recording, I first recovered the vehicle to the middle of the road and then kept it steering constantly using the mouse while driving through the curves.

This was recorded into three separate folders so as to see which improvements the additional data generated.
After the third additional recording, the model performed satisfactorily.

The final data set was augmented only by flipping the images and steering angles.

Additionally, only a third of the images where the steering angle was zero were taken into account.
This was an attempt to remove a bias towards driving straight, which was introduced during my own driving and is also apparent in the Udacity data.

The total amount of examples is around 10500 and the model was trained for 10 epochs.
This amount of data still fit into my machine's memory, so I didn't use generators here.



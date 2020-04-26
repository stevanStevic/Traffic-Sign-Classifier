## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview
---
In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I found images of traffic signs of German traffic signs on the web and put it against these 5 images.

[//]: # (Image References)

[image1]: ./examples/bar-chart.png "Visualization"
[image2]: ./examples/normalized.png "Normalized"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/new-signs.png "Traffic Signs"
[image5]: ./examples/top_5_softmax.png "Top 5"

### The Project

The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

### Data Set Summary & Exploration

First I calculated summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how occurence of given classes/labels in the dataset.

![Visualization][image1]

Values alongside x axis can be mapped to ids from `trafficsigns.csv`


### Design and Test a Model Architecture

#### Preprocessing 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Data was already scaled padded to 32x32, so I didn't have to change naything regarding this.

As a first step, I decided to normalize the images to have mean close 0, as this greatly helps the training of the model. This apporach brings stability to training and no sudden jumps in parameters (gradients). 

This is done with following code `normalized =  np.array((input - 128.0) / 128.0)`

I didin't generate additional data, however there are many methods, liek chaning colors, rotating, flipping and etc. 

Here is an example of an original image and an augmented image:

![Normalized][image2]

#### Model archicture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, same padding, outputs 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution 5x5       | 1x1 stride, same padding, outputs 10x10x32    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32				    |
| Fully connected		| Input 800, output 400 						|
| RELU					|												|
| Droput    			| 50% probability								|
| Fully connected		| Input 400, output 120 						|
| RELU					|												|
| Fully connected		| Input 120, output 84 					    	|
| RELU					|												|
| Droput    			| 50% probability								|
| Fully connected		| Input 84, output 43 					    	|
| Softmax				|           									|
|						|												|

To train the model, I used an AdamOptimizer as the most convinient for backward propagation nad cross-entrpy for error. It is sdg and aplies decay on learning rate autmoatcily. I used `EPOCHS = 25`, `BATCH_SIZE = 128`, and learning rate `rate = 0.001`

#### Solution

My final model results were:
* Training Set Accuracy  = 0.935
* Validation Set Accuracy = 0.951
* Test Accuracy = 0.935

Of course it varies from training to training. It depends on random shuffle of the data and as we saw from hisotrgam not all labels are equally represented.

With the inital LeNet I was able to reach 0.89 in few runs. From there I first started with dropouts as learned from videos that in recent years it has very good results and to prevent overfitting. I've added it after every fully connected layer. Accuracy has increased to around 0.91 and then it stagnated.

Since I added droputs my itention was to exapnd the network. I tried some convoution layer in different places and filter depths. Some brought a very small improvement and other made it even worse.

As a final solution I expanded second convo with 32 width and then flattened to bigger fully connected layer then beofre (from 400 to 800). Here I introduced dropouts and connected it to one additional fully connected layer top of this layer I added another.
I also added droputs beofre the final layer.

I tuned epoches to 25 and left inital batch size to 128.

Results on three sets of data indicated that there were no underfitting or overfitting as the results were balanced. However, a high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

There were many ways to do this with ResNets, adding skipping layer, inception module and 1x1 convolutions. On the other side not architecutral options could be, parameter tuning, using different optimizers, error functions and etc.

### Test a Model on New Images

I've found five never seen signs for the model. Downloaded them from the web and adjusted to 32x32x3 size for the model. Here are the signs

![New signs][image4]

The **'Stop'** signs has given most trouble to the model accross different training runs. It mxed it with **'No Netry'** sign as this sign has white rectangle accross and these STOP letters probably made it possible ot mix it with one line when normalized.
Second worst was the **'Hard right turn'** and other were pretty good each time.

This could also be related to how muuch these 2 signs had labels in the dataset.

#### Prediciton on new signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| No entry   									| 
| Road Work    			| Road Work 									|
| Priority road			| Priority road									|
| 70 km/h	      		| 70 km/h					 				|
|Dangerous curve to the right | Bicycles crossing      							|

The model was able to correctly guess 3 of the 5 traffic signs in this run, which gives an accuracy of 60%. It could be said that this compares favorably to the accuracy on the test set of by taking into account, that these results also depend on training run and there were cases with 4/5 and few 5/5.

Here are top 5 softmax probabilites on these signs from the notebook.

![Top 5 probs][image5]

I haven't visualied activation features from layers from the last cell in playbook, but this could be next step on this.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

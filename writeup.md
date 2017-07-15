**Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/graph.png "Visualization"
[image2]: ./writeup/input.png "Input Image"
[image3]: ./writeup/sharp_image "Sharp Image"
[image4]: ./writeup/gray_image "Gray Image"
[image5]: ./testing_data/0.png "Traffic Sign 1"
[image6]: ./testing_data/1.png "Traffic Sign 2"
[image7]: ./testing_data/2.png "Traffic Sign 3"
[image8]: ./testing_data/3.png "Traffic Sign 4"
[image9]: ./testing_data/4.png "Traffic Sign 5"
[image10]: ./testing_data/5.png "Traffic Sign 6"
[image11]: ./testing_data/6.png "Traffic Sign 7"

---
### Writeup / README

Here is a link to my [project code](https://github.com/AbhishekGurudutt/carnd-Traffic_sign_board_detection/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread across different classes.
![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to sharpen the image since image sharpening increases the contrast which helps in making the edges more defined. Here is an example of a traffic sign image before and after sharpening.

![alt text][image2] ![alt text][image3]

As a second step, I decided to convert the sharpened image to gray scale because we are detecting the traffic signs and it is dependent on the text or the pattern. The color of sign boards does not matter. Since color in the image is a noise and doesn't provide any additional feature for training, the images are converted to grayscale. Here is an example of before and after grayscaling.

![alt text][image3] ![alt text][image4]

As a last step, I normalized the image data because the range of distribution of feature values (multiplying by learning rate during back propagation) would be closer and helps to converge the gradient descent to the minimum.

**My final model consisted of the following layers:**

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs	10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten       |  output = 400            |
| Fully connected		| output = 120				|
| RELU					|												|
| Fully connected		| output = 84				|
| RELU					|												|
| Fully connected		| output = 43				|


To train the model, I used LeNet architecture which has RELU for activation. The batch size used in this project is 125 with the number of epochs to be 10 and learning rate to be 0.002.

My final model results were:
* validation set accuracy of 97.8%
* test set accuracy of 90.3%

The model used for this project is same as the one developed during lab exercise (LeNet). Initially I used the values of learning rate, epoch, and batch size to be same as in lab exercise. For preprocessing, I converted the RGB image to grayscale and normalized the image using the example equation provided. With this architecture I was able to achieve 87% accuracy.

I change the batch size from 200 to different values and finally I chose 125, I increased the learning rate from 0.001 to 0.002, and I retained the epoch value to be 10. With this change I was able to achieve an accuracy of 91%.

During the preprocessing, I sharpened the image before converting to grayscale to enhance the edges and I was able to observe a significant improvement in the accuracy (increase to 95%).

I tried to equalize the histogram, but i found that the accuracy decreased and hence I did not include histogram equalization in preprocessing.

I changed the normalization equation from ((pixel - 128) / 128) to (pixel / 255), this would help in feature distribution to be much more closer and this helped to increase the accuracy further more to 97.8%

### Test a Model on New Images

Here are few German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11]

* The first image is bright and clear and this should be classified correctly without any problem
* The second image is a bit blur.
* The third image is blur as well as tilted a bit towards the right.
* The fourth image lacks brightness.
* The fifth image is bright and clear and should not have any problem.
* The sixth image is too blurred out.
* The seventh image lacks brightness and also has reflection.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn right ahead      		| Turn right ahead   									|
| Children crossing     			| Children crossing 										|
| 30 km/h					| 30 km/h											|
| 60 km/h	      		| 60 km/h					 				|
| Stop			| Stop      							|
| 50 km/h			| 50 km/h      							|
| Keep left			| Right of way at next intersection      							|


The model was able to correctly guess 6 of 7 traffic signs, which gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 90.3%

The code for making predictions on my final model is located in the 14th and 15th cell of the Ipython notebook.

* For the first image, the model is relatively sure that this is a Turn right ahead (probability of 1.0), and the image does contain a Turn right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Turn right ahead   									|
| 0.00     				| Stop 										|
| 0.00				| Go straight or left											|
| 0.00	      			| Ahead only					 				|
| 0.00				    | Turn left ahead      							|


* For the second image the model is relatively sure that this is a Children crossing (probability of 1.0), and the image does contain a Children crossing. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Children crossing   									|
| 0.00     				| Dangerous curve to the right 										|
| 0.00				| Beware of ice/snow											|
| 0.00	      			| Road narrows on the right				 				|
| 0.00				    | Right-of-way at the next intersection				|

* For the third image the model is relatively sure that this is a 30 km/h (probability of 0.99), and the image does contain a 30 km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| 30 km/h   									|
| 0.0001     				| 80 km/h 										|
| 0.00				| 50 km/h											|
| 0.00	      			| 70 km/h				 				|
| 0.00				    | 20 km/h			|

* For the fourth image the model is relatively sure that this is a 60 km/h (probability of 0.99), and the image does contain a 60 km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| 60 km/h   									|
| 0.00005     				| 80 km/h 										|
| 0.00				| 80 km/h											|
| 0.00	      			| No Passing				 				|
| 0.00				    | 50 km/h			|

* For the fifth image the model is relatively sure that this is a Stop sign (probability of 0.99), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| Stop   									|
| 0.00     				| Turn Right ahead 										|
| 0.00				| Keep right											|
| 0.00	      			| Turn left ahead				 				|
| 0.00				    | No entry			|

* For the sixth image the model is relatively sure that this is a 50 km/h (probability of 0.99), and the image does contain a 50 km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| 50 km/h   									|
| 0.004     				| 30 km/h 										|
| 0.00001				| 80 km/h											|
| 0.00	      			| 100	km/h			 				|
| 0.00				    | Roundabout mandatory			|


* For the seventh image the model predicted to be Right of way at next intersection (probability of 0.69), but the image contain Keep left. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.69         			| Right of way at next intersection   									|
| 0.26     				| Roundabout mandatory 										|
| 0.024				| Turn right ahead											|
| 0.009	      			| Traffic signal				 				|
| 0.0008				    | Beware of ice/snow			|

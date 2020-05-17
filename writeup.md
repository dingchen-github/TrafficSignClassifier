# **Traffic Sign Recognition**

## Writeup



[//]: # (Image References)

[image1]: ./output/Label_Number.png "Label_Number"
[image2]: ./output/examples.png "examples"
[image3]: ./output/Accuracy_Orig.png "Accuracy_Orig"
[image4]: ./output/Accuracy_Orig+Dropout.png "Accuracy_Orig+Dropout"
[image5]: ./output/Accuracy_Orig+Dropout+Gray1.png "Accuracy_Orig+Dropout+Gray1"
[image6]: ./output/Accuracy_Orig+Dropout+Gray1+Norm1.png "Accuracy_Orig+Dropout+Gray1+Norm1"
[image7]: ./output/Accuracy_Orig+Dropout+Gray1+Norm2.png "Accuracy_Orig+Dropout+Gray1+Norm2"
[image8]: ./output/Label_Number_Augmented.png "Label_Number_Augmented"
[image9]: ./output/DataAugExample.png "DataAugExample"
[image10]: ./output/Accuracy_Orig+Dropout+Gray1+Norm2+DataAug.png "Accuracy_Orig+Dropout+Gray1+Norm2+DataAug"
[image11]: ./output/GPU_Accuracy_Orig+Dropout+Gray1+Norm2.png "GPU_Accuracy_Orig+Dropout+Gray1+Norm2"
[image12]: ./output/Accuracy_0.963.png "Accuracy_0.963"
[image13]: ./output/new_five.png "new_five"
[image14]: ./output/new_five_pre.png "new_five_pre"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the dataset (.p files) provided by Udacity. On the original site of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) the data exist in forms of images in folders. So the first step of arranging data is done by Udacity (Thanks!).

The pickled data is a dictionary with 4 key/value pairs, of which 2 are useful to me:
* 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
* 'labels' is a 1D array containing the label/class id of the traffic sign.

I can access the values with the following code:
```
X_training, y_train = train['features'], train['labels']
X_validation, y_valid = valid['features'], valid['labels']
X_testing, y_test = test['features'], test['labels']
```

Since the first dimension of 'features' is number of examples, the size of the training, validation and test set is easy to get with the length function:
```
n_training = len(X_training)
n_validation = len(X_validation)
n_testing = len(X_testing)
```
To get the image shape, I simply use the shape attribute:
```
image_shape = X_training[0].shape
```
To get the number of unique classes/labels, I use the set and length function. It seems that there is no need to use the pandas module.
```
n_classes = len(set(y_train))
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To get the number of each class/label, I find `np.unique()` very useful and easy.
```
label, index, count = np.unique(y_train, return_index=True, return_counts=True)
```
The variable 'label' contains arranged numbers from 0 to 42, and 'count' has the arranged number of each class. Thus I get the distribution figure of the classes.

![alt text][image1]

For every class, I print an image as an example. Some images are so dark and vague that even human eyes can not identify the traffic signs.
![alt text][image2]

---
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

At first, I run the model on my MacBook without GPU, and without preprocessing. With the LeNet5 solution from the lecture, a validation accuracy of ca. 0.89 can be achieved.
![alt text][image3]

Since the training accuracy is higher than the validation accuracy, suggesting an overfitting, I add **dropout** to the LeNet5 model between fully connected layers (note: no dropout in validation accuracy evaluation). The gap between both accuracies can be reduced, and the validation accuracy is slightly increased, meaning that dropout is a good technique.
![alt text][image4]

Then I add **grayscale** preprocessing, because in most traffic signs, color is not important information, but form. Since I use `cv2.cvtColor()`, the output shape if (32,32) instead of (32,32,1) which the model can take in, so I have to add one dimension with `np.expand_dims()` (Note: `reshape()` could do the same).

After grayscale, the accuracy curves are more stable, but there is hardly improvement of the absolute accuracy.
![alt text][image5]

Then I add **normalization** using (pixel - 128)/128. Surprisingly, the accuracy drops.
![alt text][image6]

My guess is, (pixel - 128)/128 does not give a mean of exactly 0, but slightly below 0. So I decide to try (pixel - 127.5)/127.5, which has a good result. But later we will see that this does not matter to the GPU. *(By the way, thank you Udacity for providing GPU to us!)*
![alt text][image7]

Since there is still a gap between training and validation accuracy, I decide to try **data augmentation** to add images to each class, so that all classes have the same amount of images. To generate new images, I randomly rotate and translate existing images.
```
# Rotation
rot = np.random.randint(-10, 11)
M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)

# Translation
dx, dy = np.random.randint(-4, 5, 2)
M_trans = np.float32([[1,0,dx],[0,1,dy]])

img = cv2.warpAffine(image,M_rot,(cols,rows))
img = cv2.warpAffine(img,M_trans,(cols,rows))
```
![alt text][image8]

One example of augmented image is given below.
![alt text][image9]

Add data augmentation to grayscale and normalization does close the accuracy gap, but by reducing the training accuracy, not by increasing the validation accuracy. This is the same on GPU. Considering the training is much slower with data augmentation, I decide to abandon it.
![alt text][image10]

#### 2. Describe what your final model architecture looks like.

Considering dropout and grayscale, my model consisted of the following layers:

|Layer         	|Description	        					           |
|:-------------:|:----------------------------------------:|
|Input         	|32x32x1 RGB image   							         |
|Convolution 5x5|1x1 stride, valid padding, outputs 28x28x6|
|RELU					  |												                   |
|Max pooling	  |2x2 stride, valid padding, outputs 14x14x6|
|Convolution 5x5|1x1 stride, valid padding, outputs 10x10x16|
|RELU					  |												                   |
|Max pooling	  |2x2 stride, valid padding, outputs 5x5x16|
|Flatten				|outputs 400			  		                   |
|Fully connected|outputs 120|
|RELU					  |												                   |
|Dropout					  |keep_prob = 0.5					            |
|Fully connected|outputs 84|
|RELU					  |												                   |
|Dropout					  |keep_prob = 0.5					            |
|Fully connected|outputs 43|

#### 3. Describe how you trained your model.

* Hyperparameter: After trying different configurations, I am satisfied with a batch size of 32 and a learning rate of 0.001.
* Optimizer: AdamOptimizer.
* Function to evaluate the training and validation accuracy.
* For every epoch, shuffle the training data.
* For every epoch, record the training and validation accuracy.
* Save the training model.
* Plot the curve of the training and validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I already discussed some points above in the preprocessing. I run the model on GPU and get an accuracy of ca. 0.9.
![alt text][image11]

For grayscale I used `cv2.cvtColor()`. After online research, I discovered another method:
```
np.sum(image_array/3, axis=3, keepdims=True)
```
It turns out to be a far better method. So now I have a discover that different grayscale methods could have a big impact on image training.
![alt text][image12]
After merely 30 epochs I get a validation accuracy of 0.963.

At last I run the model on the test set and get a test accuracy of 0.950.

---

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web (after resizing using `cv2.resize()`):
![alt text][image13]

Using the same preprocessing, the images look like:
![alt text][image14]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

All 5 images could be classified correctly, meaning an accuracy of 100%.

I also tried some other images and discovered that the traffic signs would better be photographed from the front, not from the side, for the model to classify them correctly. I browsed the training data for the model, indeed all images are taken from the front.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

I use the following code to get the top 5 softmax probabilities:
```
softmax_logits = tf.nn.softmax(logits)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_k = sess.run(tf.nn.top_k(softmax_logits, k=5), feed_dict={x: images_preprocessed, keep_prob : 1.0})
    print(top_k)
```

The model is 100% sure about the 1., 3. and 5. images.
The top five softmax probabilities for the 2. image were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .975         			| Speed limit (80km/h)							|
| .024     				| Speed limit (60km/h)						|
| .0008					| Speed limit (50km/h)						|
| .00006					| Speed limit (100km/h)							|
| .00002					| No passing for vehicles over 3.5 metric tons|

The top five softmax probabilities for the 4. image were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .994         			| Bumpy road							|
| .0057     				| Bicycles crossing						|
| .00031					| Traffic signals						|
| .00015					| Road narrows on the right						|
| .00008					| Dangerous curve to the right|

From the results we can see that the model can classify the traffic signs with high confidence, and the other top softmax probabilities indicate the same sort of traffic signs.

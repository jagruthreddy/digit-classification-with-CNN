# digit-classification-with-CNN
Build  a model which can identify the handwritten digits
## Table of Content
  * [Overview](#overview)
  * [Local Setup](#local-setup)
  * [Steps](#steps)

## Overview
This is a Hand Written Digit Classification with CNN which can be achieved in few steps as follows<br />
1.Import the libraries and load the dataset<br />
2.Reshaping and Normalizing the Images (preprocessing )<br />
3.Building the Convolutional Neural Network<br />
4.Compiling and training the model<br />
5.Evaluating the Model.<br />

**Dataset:** MNIST database of handwritten digits<br />
[![](https://i.imgur.com/QoWdo5o.png)](http://yann.lecun.com/exdb/mnist/)

Accuracy is above **97%**

## Local Setup
Clone the repository on your local environment <br>

```bash
git clone https://github.com/jagruthreddy/digit-classification-with-CNN
```
Navigate to the folder <br>
```bash 
cd digit-classification-with-CNN
```
USE GOOGLE COLAB OR JUPYTER NOTEBOOK<br>


## Steps
 ### 1) IMPORT DATASET 
 •Tensorflowand Kerasallow us to import and download the MNIST dataset(Modified National Institute of Standards and Technology) directly from their API<br />
 •The MNIST database contains 60,000 training images <br />
 •10,000 testing images <br />
 
 ### 2) PREPROCESS DATA
 •Reshaping the array to 4-dims so that it can work with the KerasAPI(greyscale image ) <br />
 •Need to normalize our data as it is always required in neural network models, by dividing the RGB codes to 255<br />
 •So initially will be converting data to float as we are dividing<br />
 
 ### 3) BUILDING THE CONVOLUTIONAL NEURAL NETWORK
 **Sequential model** allows you to build a model layer by layer.model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))#28 number of layers<br />
 **Max pooling** is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map<br />
 **Flattening** involves transforming the entire pooled feature map matrix into a single column which is then fed to the neural network for processing.<br />
 **DENSE LAYER :** A linear operation in which every input is connected to every output by a weight . It connects neurons in one layer to neurons in another layer. It is used to classify images between different category by training.<br />
 **DROPOUT :** A Simple Way to Prevent Neural Networks from Overfitting<br />
 **SOFTMAX  LAYER** is the last layer of CNN. It resides at the end of FC layer, softmaxis for multi-classification.<br />

### 4) COMPILING AND TRAINING THE MODEL 
Compiling the model takes three parameters: **optimizer, loss and metrics.** <br />
Optimizer: controls the learning rate. We will be using ‘adam’ as our optmizer. Adam is generally a good optimizer to use for many cases. The adamoptimizer adjusts the learning rate throughout trainingLoss: that can be used to estimate the loss of the model so that the weights can be updated to reduce the loss on the next evaluation.We use fit method then training starts <br />
### 5) EVALUATING THE MODEL 
With a simple evaluate function to know the accuracy 
## Technologies Used
![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1915px-Tensorflow_logo.svg.png" width=160>](https://www.tensorflow.org/) [<img target="_blank" src="https://seeklogo.com/images/M/matplotlib-logo-7676870AC0-seeklogo.com.png" width=160>](https://matplotlib.org/)

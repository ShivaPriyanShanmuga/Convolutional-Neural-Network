**Convolutional Neural Network**

**Image Recognition using Custom CNN in NumPy and Keras**

This project implements an image recognition system using a custom Convolutional Neural Network (CNN) built from scratch with NumPy and the CIFAR-10 dataset. The code manually implements layers and functions such as convolution, ReLU activation, batch normalization, max pooling, and softmax classification.

**Dataset**

The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is automatically downloaded via the keras.datasets.cifar10 module.

**Features**

Manual CNN Construction: Implements convolution, batch normalization, ReLU activation, max pooling, flattening, and softmax without deep learning libraries.

Data Preprocessing: Normalization and one-hot encoding.

Training and Testing: Separate functions for training and testing the CNN.

**Requirements**

Python 3.x

NumPy

Keras

**Training Parameters**

Epochs: 10

Batch Size: 32

Learning Rate: 0.001

Train Dataset Size: 10,000 samples

**Results**

The script will print the training accuracy and loss after each epoch, followed by the test accuracy.

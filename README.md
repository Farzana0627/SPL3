# Speech Recognition for Bengali Digits

Speech Recognition is one of the important and challenging tasks in today’s world. Though much effort has been made to gain maximum accuracy level for many languages, there is still room for improvement in the recognition task of Bengali speech. Keeping this in mind, an initiative has been taken in this work for recognition of Bengali digits (from 0 to 9). Due to time and resource constraints, a small vocabulary set has been chosen. RPLPwavelet, a modified version of feature extraction method Perceptual Linear Prediction has been implemented and trained with CNN (Convolutional Neural Network). Dataset containing 1000 audio files uttering 0 to 9 in Bengali having 100 files per digit class has been prepared. The audio files were recorded in different noisy environments such as classroom, streets, crowded home environment etc. Significant classification accuracy has been achieved from testing the generated model with different dataset.

## Implementation Detail

###Feature Extraction

The sound features for the methods PLP, RPLP and RPLPwavelet are extracted using Matlab codes. Rastamat [27] sound processing toolbox is used in this work for calculating the PLP cepstral coefficients. 

The 'Feature Extraction' folder contains implementation of speech feature extraction for PLP, RPLP and RPLPwavelet in three different ways. In all the folders there is a file named 'Main.m'. Running this file in matlab will generate either train or test speech files' features in individual folders in the same directory. Suppose in 'PLPusingrasta', there are two generated folders 'PLPTrainResultFolder' and 'PLPTestResultFolder'.

###Feature Classification

Acoustic features are obtained from application of PLP, RPLP and RPLPwavelet. Since CNN (Convolutional Neural Network) demands large dataset to train, artificial data are synthesized to be used for training the neural network model along with the authentic ones. Before training the model, the data need to be normalized. In this work, normalization with mean=zero and standard deviation=1 is performed. Keras library from Python is used to train the CNN model here that uses Tensorflow at its backend. Due to gpu support of tensorflow, training the models is very time efficient now. A dataset containing 500 samples per digit class is trained with two convolutional layers containing kernel size 128 and two dense layers with kernel size 200 in its first layer are applied. In the first three layes activation function ‘relu’ is used, and ‘softmax’ is used in the last layer. Different strategies and combinations are followed for obtaining the best results of each of the method. The generated models are saved as .json format for future loading and testing with different test datasets.

In the folder named 'Training', there is a python file named 'cnn_final.py'. Running this file along with the generated features from feature extraction phase will generate validation accuracy for given number of epochs in Convolutional Neural Network. Here in this research work, NVIDIA gpu has been used to make the training faster and it could achieve % accuracy using the method RPLPwaveletusingrastamodified3.


### Prerequisites

1. Matlab 2014 or above with speech recognition toolbox.
2. Python version 3.5 or above
3. Tensorflow gpu version 1.4.1 if NVIDIA gpu is used/ Tensorflow CPU version 1.4.1
4. Keras library
5. Python libraries (numpy, imblearn, hd5y)


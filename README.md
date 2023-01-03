# Image captioning using attention-based model and non attention_based model


## Table of Contents
1. [Description](#Description)
2. [Objective](#Objective)
3. [Dataset](#Dataset)
4. [Preprocessing](#Preprocessing)
5. [ML_Pipeline](#ML_Pipeline)
6. [Installation](#Installation)
7. [How to run](#How-to-run)


### Description
In this project I tried to build a model that can be used in captioning images as image captioning is an important task that we can use AI to facilitate it. 


### Objective 
The aim of this project for me is to learn how to deal with the image captioning tasks and understand the attention technique so the results may not be very good but i will try to improve it later. In this project I learned how to train and validate a model in image captioning tasks using pytorch framework. I learned the following:
* build a custom dataset class
* which loss function should be used (Cross Entropy)
* how to evaluate the model using BLEU score
* understand RNNs such as LSTM architecture
* understand attention technique and how to implement it


### Dataset
The msCOCO dataset consists of both images and thier annotaions.

The training, validation and testing datasets consist of 83K, 41K and 41K images respectively. Images have different sizes.


### Preprocessing
* resizing the images into unified size (224*224) because my pc is not strong enough to deal with larger size but it's better to increase the size
* normalize the images


### ML_Pipeline
* create a custom dataset class
* apply some data augmentation techniques (random rotation, random translation, random scaling and random cropping)
* use dataloaders
* create model consists of CNN-encoder (resnet50) and RNN-decoder (LSTM architecture)

I didn't train the models for enough number of epochs because the dataset is huge and my pc is not strong enough so we need to continue the training process as the model doesn't overfit yet.


### Installation
You must have the latest versions(October 2022) of these libraries:
* numpy
* matplotlib
* tqdm
* pytorch
* json
* PILLOW

And of course you should have python 3.8.5 and jupyter


### How to run
Follow these steps to run the project:
If you want to use non-attention-based model:
1. run the training.ipynb
2. run inference.ipynb

But if you want to use attention-based model:
1. run the training attention.ipynb
2. run inference attention.ipynb
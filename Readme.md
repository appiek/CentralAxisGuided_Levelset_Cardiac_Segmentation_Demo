## Overview
We present a novel and efficient computing framework for segmenting the myocardium from Cardiac MRI by combining deep convolutional neural network with proposed central axis guided level set approach.
We implemented our method based on the open-source machine learning framework TensorFlow and reinforcement learning library TensorLayer. This repository contains all code used in our experiments, incuding the data preparation, model construction, model training, model inferencing and result evaluation. 

## Dependencies  
* Matlab
* Python 3.6
* TensorFlow 1.10
* TensorLayer 1.9.1
* Scikit-image 14.0
* Numpy
* Scipy

## Dataset
We utilized the training set of MICCAI 2013 SATA Challenge to train our model. And we evalulated our method on MICCAI 2009 LV segmentation dataset.  
* MICCAI 2013 SATA Challenge: includes totally 83 annotated NII image [Link](http://www.cardiacatlas.org/data-access/)
* MICCAI 2009 LV segmentation dataset: contains 45 manually annotated Cardiac MRI cases [Link]( https://smial.sri.utoronto.ca/LV Challenge/)
 
## Composition of code
1. the main steps for data preparation, model training and result evaluation:
    * step_1: randomly extracting the image patches from original data 
    * step_2: randomly divide the image patches as training and validation data
    * step_3: transforming the image patches into tfrecord file
    * step_4: training the proposed convolutional neural network-based myocardial central axis model
    * step_5: using the networks to segment the testing images
    * step_6: evaluating the segmentation results 

2. ./tools: image patches extraction and weight map generation
3. ./nets: model construction
4. ./utils: producing tfrecord file and image post-processing
5. ./Evaluation Metrics: evaluation methods
6. ./loaddata: loading the NII data by Matlab

## Quick Start
* Testing: if you just want to validate the segmentation performance of pre-trained models, follow these steps:
   1. Download our code on your computer, assume the path is "./";
   2. Download the MICCAI 2009 LV segmentation dataset and unzip this file in your computer
   3. Download the pre-trained parameters of model [Link]() and unzip this file into the path './checkpoints/'
   4. Open the file 'step5_segmentatioan_batchnii_CGLS.py', modify the parameter 'data_path'. Then run the code for segmenting the testing images. 
   5. Run 'step6_evaluation.m' for evaluating the performance of method

## Contact information  
* E-mail: xlpflyinsky@foxmail.com

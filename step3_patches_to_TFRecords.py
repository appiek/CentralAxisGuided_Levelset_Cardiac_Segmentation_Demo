# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:34:41 2017

@author: XLP
"""

from utils.write_read_tfrecord import *
import os.path as osp
import tensorflow as tf 
from PIL import Image
import matplotlib.pyplot as plt



#=========================step1: Set parameters===============================
datapath = './data/'  #the path to store patches and lmdb
save_path = './data/'
npatches = 1
nfolds = 1
#patchpathlist = glob.glob(datapath+'patches_'+'*')

data_shape = [224,224]
#=========================step2: Creat tfrecord===================================
for i in range(1,npatches+1,1):
    imgpath = datapath + 'training-set_'+str(i)
    for j in range(1,nfolds+1,1):
        train_datalist = osp.join(imgpath,'patches_train_'+str(j)+'.txt')
        valid_datalist = osp.join(imgpath,'patches_valid_'+str(j)+'.txt')
        train_tfrecord = osp.join(save_path,'train_'+str(i)+'_'+str(j)+'.tfrecords')
        valid_tfrecord = osp.join(save_path,'valid_'+str(i)+'_'+str(j)+'.tfrecords')   
        write_images_tfrecord(train_datalist,imgpath,train_tfrecord,data_shape,True)
        write_images_tfrecord(valid_datalist,imgpath,valid_tfrecord,data_shape)

img, label, weight = read_and_decode(train_tfrecord,data_shape)      

# apply "shuffle_batch" to load data randomly
img_batch, label_batch, weight_batch = tf.train.shuffle_batch([img, label, weight],
                                                batch_size=10, capacity=1000,
                                                min_after_dequeue=500)

#=========================step3: valid tfrecord===================================
init = tf.global_variables_initializer()
coord = tf.train.Coordinator() #stop thread
with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(10):
        val, l, fw= sess.run([img_batch, label_batch, weight_batch])
    
        #l = to_categorical(l, 12) 
        print(val.shape)
    coord.request_stop() #stop thread


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:14:18 2017

@author: XLP
"""
import tensorflow as tf 
from PIL import Image
import numpy as np
import scipy.io as sio 

def write_images_tfrecord(imglistpath,imgpath,tfreconame,data_shape,opt_meanval=False):
    writer = tf.python_io.TFRecordWriter(tfreconame)
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    fp = open(imglistpath,"r")  
    lines = fp.readlines()#读取全部内容   
    N = len(lines)	# Number of data instances

    if opt_meanval:
        meanvalname = tfreconame+'_mean.npy'
        meanval = np.zeros(data_shape)
        for i in range(N):
            imgname = lines[i][0:-1]
            ##=========load and write image, label data, foreground_weight
            img = Image.open(imgpath + '/' + imgname + '.png')    
            label =  Image.open(imgpath + '/' + imgname + '_l.png')            
            ## sum all data
            meanval = meanval + np.array(img)
            img = img.tobytes()
            label = np.array(label.convert('L'),'uint8')
            label = label.tobytes()
            foreground_weight = sio.loadmat(imgpath + '/' + imgname + '_w.mat')
            foreground_weight = foreground_weight['patch_weight']
            foreground_weight= foreground_weight.tobytes()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])), 
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                'foreground_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[foreground_weight])),
            }))
            writer.write(example.SerializeToString())  #序列化为字符串 
        meanval = meanval / N
        meanval = np.mean(np.reshape(meanval,[data_shape[0]*data_shape[1]]),0)
        np.save(meanvalname,meanval)
    else:
        for i in range(N):
            imgname = lines[i][0:-1]
            ##=========load and write image, label data, foreground_weight
            img = Image.open(imgpath + '/' + imgname + '.png')    
            label =  Image.open(imgpath + '/' + imgname + '_l.png')            
            img = img.tobytes()
            label = np.array(label.convert('L'),'uint8')
            label = label.tobytes()
            foreground_weight = sio.loadmat(imgpath + '/' + imgname + '_w.mat')
            foreground_weight = foreground_weight['patch_weight']
            foreground_weight= foreground_weight.tobytes()
           
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])), 
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                'foreground_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[foreground_weight])),
            }))
            writer.write(example.SerializeToString())  #序列化为字符串 

    writer.close() 
    fp.close()

def read_and_decode(tfreconame,data_shape):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([tfreconame])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                      features={
                                           'image':tf.FixedLenFeature([],tf.string), 
                                           'label':tf.FixedLenFeature([],tf.string),
                                           'foreground_weight':tf.FixedLenFeature([],tf.string),
                                       })
    
    img = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)
    img = tf.reshape(img, data_shape)
    img = tf.cast(img, tf.float32)
    label = tf.reshape(label, [data_shape[0], data_shape[1]])
    label = tf.one_hot(indices=label,depth=2)   
    label = tf.cast(label, tf.float32)
    foreground_weight = tf.decode_raw(features['foreground_weight'], tf.float64)   
    foreground_weight = tf.reshape(foreground_weight, [data_shape[0], data_shape[1]])
    foreground_weight = tf.cast(foreground_weight, tf.float32) 
    
     
    return img, label, foreground_weight
    

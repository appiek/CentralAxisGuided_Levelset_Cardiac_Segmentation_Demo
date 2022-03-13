# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:36:50 2017

@author: XLP
"""
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import list_remove_repeat
from tensorlayer.layers.core import LayersConfig
                    
   
Layer = tl.layers.Layer
def image_preprocess(img,meanval):
    meanval = tf.constant(meanval,tf.float32)
    img = tf.cast(img, tf.float32) 
    img = tf.subtract(img, meanval)* (1./255)
    return img

class Mergelayer(Layer):
    def __init__(
        self,
        layer = [],
        name ='merge_layer',
    ):
        Layer.__init__(self, prev_layer = None, name=name)
        
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)    
  
    
class ChannelAttentionlayer(Layer):
  def __init__(
      self,
      layer = None,
      output_dim = 128,
      ratio = 2,
      name ='channelatten_layer',
  ):
      # check layer name (fixed)
      Layer.__init__(self, prev_layer=layer, name=name)

      # the input of this layer is the output of previous layer (fixed)
      self.inputs = layer.outputs
      # print out info
      print("   ChannelAttentionlayer %s: output_dim %d, ratio %s" % (self.name, output_dim, ratio))
      # operation (customized)
      with tf.variable_scope(name):
          # create new parameters
          squeeze = tf.reduce_mean(self.inputs, axis=[1, 2],name=name+'_Global_Mean')
          #======================dense connected1 =======================
          n_in1 = int(squeeze.get_shape()[-1])
          W1 = tf.get_variable(
                name='W1', shape=(n_in1, int(output_dim/ratio)), initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=LayersConfig.tf_dtype)
          b1 = tf.get_variable(name='b1', shape=(int(output_dim/ratio)), initializer=tf.constant_initializer(value=0.0), dtype=LayersConfig.tf_dtype)
          excitation = tf.nn.bias_add(tf.matmul(squeeze, W1), b1, name='bias_add1')
          excitation = tf.nn.relu(excitation)
          #=====================dense connected2===========================
          n_in2 = int(excitation.get_shape()[-1])
          W2 = tf.get_variable(
                name='W2', shape=(n_in2, output_dim), initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=LayersConfig.tf_dtype)
          b2 = tf.get_variable(name='b2', shape=(output_dim), initializer=tf.constant_initializer(value=0.0), dtype=LayersConfig.tf_dtype)
          excitation = tf.nn.bias_add(tf.matmul(excitation, W2), b2, name='bias_add2')
          excitation = tf.nn.sigmoid(excitation)
          
          excitation = tf.reshape(excitation, [-1,1,1,output_dim])
          # tensor operation
          self.outputs = self.inputs* excitation

      # update layer (customized)
      self.all_layers.extend( [self.outputs])
      self.all_params.extend( [W1, b1, W2, b2])            
                                
def model_VGG16_Landmark_Detection(x_ori,y_,fw, batch_size,data_shape,reuse, mean_file_name=None,is_train = True):
    if mean_file_name!=None:
        meanval = np.load(mean_file_name)
        x = image_preprocess(x_ori,meanval)
    else:
        x = x_ori
    drop_rate = 0.6 
    sida = 0.000001
    up_dim = 16
    gamma_init=tf.random_normal_initializer(2., 0.01)
    with tf.variable_scope("VGG16_Deep_RSF", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network_input = tl.layers.InputLayer(x, name='input')
        """ conv1 """
        network1 = tl.layers.Conv2dLayer(network_input, shape = [3, 3, data_shape[2], 64],
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
        network1 = tl.layers.Conv2dLayer(network1, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
        network1 = tl.layers.BatchNormLayer(network1, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        network1 = tl.layers.PoolLayer(network1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', 
                    pool = tf.nn.max_pool, name ='pool1') #outputsize: [H/2,W/2]
        """ conv2 """
        network1 = tl.layers.DropoutLayer(network1,keep=drop_rate,is_train=is_train,name='drop_net1')
        network2 = tl.layers.Conv2dLayer(network1, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
        network2 = tl.layers.Conv2dLayer(network2, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
        network2 = tl.layers.BatchNormLayer(network2, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        network2 = tl.layers.PoolLayer(network2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool2') #outputsize: [H/4,W/4]
        """ conv3 """
        network2 = tl.layers.DropoutLayer(network2,keep=drop_rate,is_train=is_train,name='drop_net2')
        network3 = tl.layers.Conv2dLayer(network2, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
        network3 = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
        network3 = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
        network3 = tl.layers.BatchNormLayer(network3, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        network3 = tl.layers.PoolLayer(network3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool3') #outputsize: [H/8,W/8]
        """ conv4 """
        network3 = tl.layers.DropoutLayer(network3,keep=drop_rate,is_train=is_train,name='drop_net3')
        network4 = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
        network4 = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
        network4 = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
        network4 = tl.layers.BatchNormLayer(network4, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        network4 = tl.layers.PoolLayer(network4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool4') #outputsize: [H/16,W/16]
        """ conv5 """
        network4 = tl.layers.DropoutLayer(network4,keep=drop_rate,is_train=is_train,name='drop_net4')
        network5 = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
        network5 = tl.layers.Conv2dLayer(network5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
        network5 = tl.layers.Conv2dLayer(network5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
        network5 = tl.layers.BatchNormLayer(network5, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn5')
        network5 = tl.layers.PoolLayer(network5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool5') #outputsize: [H/32,W/32]
        network5 = tl.layers.DropoutLayer(network5,keep=drop_rate,is_train=is_train,name='drop_net5')
        '#########################Upsample and merge##########################'
        '''top-down 5'''
        network5_conv = tl.layers.Conv2dLayer(network5, shape = [3, 3, 512, up_dim], act = tf.nn.relu,  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv5_conv')
        network5_up = tl.layers.UpSampling2dLayer(network5_conv,
                    size = [data_shape[0]//16,data_shape[1]//16],method =0,is_scale = False,name = 'upsample5' )   # output:[H/16,W/16,64]
        '''top-down 4'''
        network4_conv = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, up_dim], act = tf.nn.relu, # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_conv')
        network_cmb4_5 = tl.layers.ConcatLayer([network4_conv,network5_up],
                    concat_dim = 3, name = 'concat_4_5') # output:[H/16,W/16,64]
        network4_up = tl.layers.UpSampling2dLayer(network_cmb4_5, 
                    size = [data_shape[0]//8,data_shape[1]//8], method =0,is_scale = False,name = 'upsample4' )   # output:[H/8,W/8,128]
        '''top-down 3'''
        network3_conv = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, up_dim], act = tf.nn.relu, # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv3_conv') # output:[H/8,W/8,64]
        network_cmb3_4 = tl.layers.ConcatLayer([network3_conv,network4_up],
                    concat_dim = 3, name = 'concat_3_4')# output:[H/8,W/8,96]
        network3_up = tl.layers.UpSampling2dLayer(network_cmb3_4, 
                    size = [data_shape[0]//4,data_shape[1]//4], method =0,is_scale = False,name = 'upsample3' )   # output:[H/4,W/4,192]
        '''top-down 2'''
        network2_conv = tl.layers.Conv2dLayer(network2, shape = [3, 3, 128, up_dim],act = tf.nn.relu,  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv2_conv') # output:[H/4,W/4,64]
        network_cmb2_3 = tl.layers.ConcatLayer([network2_conv,network3_up],
                    concat_dim = 3, name = 'concat_2_3')# output:[H/4,W/4,128]
        network2_up = tl.layers.UpSampling2dLayer(network_cmb2_3, 
                    size = [data_shape[0]//2,data_shape[1]//2], method =0,is_scale = False,name = 'upsample2' )   # output:[H/2,W/2,256]

        '''top-down 1'''
        network1_conv = tl.layers.Conv2dLayer(network1, shape = [3, 3, 64, up_dim], act = tf.nn.relu, # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_conv') # output:[H/2,W/2,64]
        network_cmb1_2 = tl.layers.ConcatLayer([network1_conv,network2_up],
                    concat_dim = 3, name = 'concat1_2')# output:[H/2,W/2,160]
        network1_up = tl.layers.UpSampling2dLayer(network_cmb1_2, 
                    size = [data_shape[0],data_shape[1]], method =0,is_scale = False,name = 'upsample1' )   # output:[H,W,320]
        
                                                                           
        
        """## cost of classification3"""
        network1_up = ChannelAttentionlayer(network1_up, output_dim=up_dim*5,ratio=2,name='CA1')
        network1_up = tl.layers.DropoutLayer(network1_up,keep=drop_rate,is_train=is_train,name='up1')
        network_class3 = tl.layers.Conv2dLayer(network1_up,             
               shape = [3, 3, up_dim*5, up_dim*5//2], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='score1_feaconv')  # output: (?, 14, 14, 64)
        network_class3 = tl.layers.DropoutLayer(network_class3,keep=drop_rate,is_train=is_train,name='drop3')
        network_class3 = tl.layers.Conv2dLayer(network_class3,             
               shape = [3, 3, up_dim*5//2, 2], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='output')  # output: (?, 14, 14, 64)
			   
        if is_train:     
            #================cost1================================
            y = network_class3.outputs   
            y_prob = tf.nn.softmax(y,3)  
            y_class = tf.argmax(y_prob,3)    
            y_class = tf.reshape(y_class,[batch_size, data_shape[0],data_shape[1],1])              
            cost = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y_prob+sida),3),fw))
            #================costall================================
            return network_class3,cost,y_class
        else:
            y = network_class3.outputs   
            y_prob = tf.nn.softmax(y,3)
#            y_class = tf.argmax(y_prob,3)
            y_prob = tf.reshape(y_prob,[batch_size, data_shape[0],data_shape[1],2])
            return network_class3,y_prob
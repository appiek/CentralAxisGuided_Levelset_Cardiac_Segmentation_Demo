# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:03:21 2018

@author: Xie Lipeng
"""

import numpy as np
from utils.write_read_tfrecord import *
import tensorflow as tf
import tensorlayer as tl
import time
from nets.construct_model_landmarkdetection_networks import *
import os
import matplotlib.pyplot as plt

#=========================step1: Set parameters===============================
datapath = './data/'  #the path to store patches and lmdb
save_path = './data/'
model_path = './checkpoints/'
if not os.path.exists(model_path):
    os.mkdir(model_path)


npatches = 1
nfolds = 1
batch_size = 4
#patchpathlist = glob.glob(datapath+'patches_'+'*')
model_file_name = model_path + "model_tfrecord_279.ckpt"
option_resume = True # load model, resume from previous checkpoint?
option_savenet = True
data_shape = [224,224,1]


train_tfrecord = './data/train_1_1.tfrecords'
valid_tfrecord = './data/valid_1_1.tfrecords'
mean_file_name = './data/train_1_1.tfrecords_mean.npy'
is_train = True
reuse = False
trainnum = sum(1 for _ in tf.python_io.tf_record_iterator(train_tfrecord)) 
validnum = sum(1 for _ in tf.python_io.tf_record_iterator(valid_tfrecord)) 

min_after_dequeue_train = trainnum
capacity_train = min_after_dequeue_train + 3 * batch_size
capacity_test = validnum + 3 * batch_size

#=========================step2: Load Dat====a================================
train_img, train_label, train_weight = read_and_decode(train_tfrecord,data_shape)      
#使用shuffle_batch可以随机打乱输入
X_train, y_train, fw_train = tf.train.shuffle_batch([train_img, train_label,train_weight],
                                                batch_size=batch_size, capacity=1000,
                                                min_after_dequeue=500)

#valid_img, valid_label,valid_weight  = read_and_decode(valid_tfrecord,data_shape)      
##使用shuffle_batch可以随机打乱输入
#X_valid, y_valid, fw_valid = tf.train.batch([valid_img, valid_label,valid_weigh],
#                                                batch_size=batch_size, capacity=100)

#=========================step3: Creat network===============================
sess = tf.Session()

# Define the batchsize at the begin, you can give the batchsize in x and y_
# rather than 'None', this can allow TensorFlow to apply some optimizations
# – especially for convolutional layers.
x = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 1])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1], 2])
fw = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1]])

#_, cost_test,acc_test = model(x,y_,w,reuse=True,is_train=False)
                        
#=========================step4: Train network===============================
# train
n_epoch = 200
n_step_epoch = int(trainnum/batch_size)
n_step = n_epoch * n_step_epoch

learning_rate = 0.00001
print_freq = 20

network,cost,op_class1 = model_VGG16_Landmark_Detection(x,y_,fw, batch_size,data_shape,reuse=reuse,mean_file_name=mean_file_name, is_train = is_train)
#_,op_class_valid = model_VGG16_Landmark_Detection(x,y_,fw,batch_size,data_shape,reuse=True,mean_file_name=mean_file_name, is_train = False)
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
#
#train_op = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, 
#rho=0.95, epsilon=1e-08).minimize(cost, var_list=train_params)

#tl.layers.initialize_global_variables(sess)
init=tf.global_variables_initializer()  
sess.run(init)

if option_resume:
    print("Load existing model " + "!"*10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)

network.print_params(False)
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)

coord = tf.train.Coordinator() #stop thread
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
step = 0
for epoch in range(n_epoch):
    start_time = time.time()
    train_acc, train_loss,  n_batch = 0, 0, 0
    for s in range(n_step_epoch):
        X_train_a, y_train_a, fw_train_a = sess.run([X_train, y_train,fw_train])
        feed_dict = {x: X_train_a, y_: y_train_a, fw:fw_train_a}
        feed_dict.update(network.all_drop)   # enable noise layers
        _, err, output_class_train= sess.run([train_op,cost,op_class1], feed_dict=feed_dict)
        step += 1
        train_loss += err
        n_batch += 1

        
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d : Step %d-%d of %d took %fs" % (epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train Dice acc: %f" % (train_acc/ n_batch))
        tl.visualize.images2d(images=output_class_train*255, second=0, saveable=True, name='images1_'+str(epoch), dtype=None, fig_idx=3119362)
        #==================test=========================
#        valid_acc, n_batch = 0, 0
#        for _ in range(int(validnum/batch_size)):
#            X_valid_a, y_valid_a= sess.run([X_valid, y_valid])
#            feed_dict = {x: X_valid_a, y_: y_valid_a}
#            output_class_valid  = sess.run(op_class_valid, feed_dict=feed_dict)
#            output_class = output_class_valid[:,:,:,0]<=0
#            valid_acc += Dice(y_valid_a[:,:,:,1]>0,output_class); n_batch += 1
#        print("   Valid Dice acc: %f" % (valid_acc/ n_batch))

        
        if option_savenet:
            print("Save model " + "!"*10)
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_path+"model_tfrecord_"+str(epoch)+".ckpt")

        
coord.request_stop() #stop thread
coord.join(threads)
sess.close()
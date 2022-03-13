# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:09:41 2019

@author: dlwork
"""

import numpy as np
from utils.write_read_tfrecord import *
import tensorflow as tf
import tensorlayer as tl 
from PIL import Image
import matplotlib.pyplot as plt
import glob
import scipy.misc
import os
import nibabel as nib
from nets.construct_model_landmarkdetection_networks import *
from skimage import morphology 
from scipy import ndimage as ndi
from scipy import signal
import cv2
import scipy.io as sio
import pydicom
from skimage.measure import find_contours
import  skimage.measure as measure
import time
from skimage.segmentation import find_boundaries

def Detection_Result_PostProcess(MaskP):
    Mask_max = save_max_objects(MaskP)
    Mask_convx = morphology.convex_hull_object(Mask_max)
    Mask_convx_Hole = np.logical_and(Mask_convx,np.logical_not(Mask_max))
    if np.sum(Mask_convx_Hole)<=0:
        return morphology.skeletonize(Mask_max)
    Mask_convx_max = save_max_objects(Mask_convx_Hole)
    
    # merge the hole and the Mask_max
    Mask_mixture = np.logical_or(Mask_convx_max,Mask_max)
    Mask_mixture_outer =  np.logical_and(np.logical_not(morphology.erosion(Mask_mixture,morphology.square(7))),Mask_mixture)
    Mask_conn = np.logical_or(Mask_max,np.logical_and(np.logical_not(Mask_max),Mask_mixture_outer))
    
    marker_mixture_contour = morphology.skeletonize(Mask_conn)
    [idx,idy] = np.nonzero(marker_mixture_contour)
    idx = np.int32(np.mean(idx))
    idy = np.int32(np.mean(idy))
    
    Mask_fill = morphology.flood_fill(np.int32(marker_mixture_contour),(idx, idy), 1, connectivity=1)
    Mask_inner = np.logical_and(Mask_fill,np.logical_not(marker_mixture_contour))
    marker = find_boundaries(Mask_inner,mode='outer')
    
    Mask_ca = morphology.dilation(marker,morphology.square(3))
    return Mask_ca

def save_max_objects(img):
    labels = measure.label(img)  
    jj = measure.regionprops(labels)  
    # is_del = False
    if len(jj) == 1:
        out = img
        # is_del = False
    else:
        num = labels.max()  
        del_array = np.array([0] * (num + 1))
        for k in range(num):
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  
            else:
                k_area = jj[k].area  
                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
        # is_del = True
    return out


def show_curve_and_phi(fig, I, phi, color):
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray')
    fig.axes[0].contour(phi, 0, colors=color,alpha=1, linewidths=0.2,linestyles='solid')
    fig.axes[0].set_axis_off()
    plt.draw()



def Dice_Jaccard_ND(G,S):
    GS_Inter = G & S
    GS_Inter = np.sum(GS_Inter)
    GS_Union = G | S
    GS_Union = np.sum(GS_Union)
    
    G = np.sum(G)
    S = np.sum(S)
    Dice = 2*GS_Inter/(G+S)
    Jaccard  = GS_Inter/GS_Union               
    return Dice, Jaccard

def F1score_ND(G,S):
    
    TP = np.sum(G & S)
    FP = np.sum(np.logical_not(G)&S)
    
    Prec = TP/(TP+FP)
    Rec = TP/np.sum(G)
    F1_s = 2*Prec*Rec/(Prec+Rec)
    return Prec,Rec,F1_s

def Pixel_evalution_ND(G,S):
    T1 = np.sum(G&S)  
    T0 = np.sum(np.logical_not(G)&np.logical_not(S)) 
    P = T1/np.sum(G)
    Q = T0/np.sum(np.logical_not(G))
    PPV = T1/np.sum(S)
    NPV = T0/np.sum(np.logical_not(S))
    return P,Q,PPV,NPV  
    
def MaxMin_normalization(I):
    Maxval = np.max(I)
    Minval = np.min(I)
    II=np.float32((I-Minval)/(Maxval-Minval)*255)
    return II

def save_contours(phi,img_name,save_path):    
    contours = find_contours(phi, 0)
    lengthc = []
    bais_value = 1
    for n, contour in enumerate(contours):
        lengthc.append(len(contour[:, 1]))
       
    numc = len(contours)    
    if numc>=2:
        idx = np.argsort(-np.array(lengthc))
        ocontour = np.transpose(np.float32(contours[idx[0]]))
        icontour = np.transpose(np.float32(contours[idx[1]]))
        
        
        txtName = save_path+'/'+img_name+"-ocontour-auto.txt"
        f = open(txtName,'w+')
        for i in range(0,len(ocontour[1,:])):
            f.writelines('%.2f %.2f\n' %(ocontour[1,i]+bais_value,ocontour[0,i]+bais_value))
        f.close()
    
        txtName = save_path+'/'+img_name+"-icontour-auto.txt"
        f = open(txtName,'w+')
        for i in range(0,len(icontour[1,:])):
            f.writelines('%.2f %.2f\n' %(icontour[1,i]+bais_value,icontour[0,i]+bais_value))
        f.close()
    elif numc==1:
        ocontour = np.transpose(np.float32(contours[0]))
        txtName = save_path+'/'+img_name+"-ocontour-auto.txt"
        f = open(txtName,'w+')
        for i in range(0,len(ocontour[1,:])):
            f.writelines('%.2f %.2f\n' %(ocontour[1,i]+bais_value,ocontour[0,i]+bais_value))
        f.close()
        
        icontour = np.transpose(contours[0])
        txtName = save_path+'/'+img_name+"-icontour-auto.txt"
        f = open(txtName,'w+')
        for i in range(0,len(icontour[1,:])):
            f.writelines('%.2f %.2f\n' %(icontour[1,i]+bais_value,icontour[0,i]+bais_value))
        f.close()

def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 0):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky)) 

def NeumannBoundCond(f):
    nrow,ncol = np.shape(f)
    g=f
    g[[0,nrow-1],[0,ncol-1]]=g[[2,nrow-3],[2,ncol-3]]
    g[[1, nrow-1],1:-2] = g[[2,nrow-3],1:-2]   
    g[1:-2,[0,ncol-1]] = g[1:-2,[2, ncol-3]]
    
    return g
  
def div(Nx,Ny):                       
    [nxx,junk] = np.gradient(Nx)                          
    [junk,nyy] = np.gradient(Ny)                            
    k = nxx+nyy
    return k

def Del2(u):
    Lg = np.array([0,0.25,0,0.25,0,0.25,0,0.25,0])
    Lg = np.reshape(Lg,[3,3])
    L =  signal.convolve2d(u,Lg,mode='same')
    k = L-u
 
    return k
    
def distReg_p2(phi):
    # compute the distance regularization term with the double-well potential p2 in eqaution (16)
    phi_x,phi_y = np.gradient(phi)
    s = np.sqrt(phi_x**2 + phi_y**2)
    a = (s>=0) & (s<=1)
    b= s>1
    ps=a*np.sin(2*np.pi*s)/(2*np.pi)+b*(s-1)  # compute first order derivative of the double-well potential p2 in eqaution (16)
    dps=((ps!=0)*ps+(ps==0))/((s!=0)*s+(s==0))  # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    f = div(dps*phi_x - phi_x, dps*phi_y - phi_y) + 4*Del2(phi) 
    return f

def Dirac(x, sigma):
    f=(0.5/sigma)*(1+np.cos(np.pi*x/sigma))
    b = (x<=sigma) & (x>=-sigma)
    f = f*b
    return f
    
def MarkGuided_LevelSet(phi_0, g, DistanceMap, lambda_p, mu, alfa, v, epsilon, timestep, iter, potentialFunction):
    phi=phi_0;
    vx, vy=np.gradient(g)
    smallNumber=1e-10
    for k in range(iter):
        phi=NeumannBoundCond(phi)
        phi_x,phi_y = np.gradient(phi)
        s=np.sqrt(np.square(phi_x) + np.square(phi_y))
        Nx=phi_x/(s+smallNumber)# add a small positive number to avoid division by zero
        Ny=phi_y/(s+smallNumber)
        curvature=div(Nx,Ny)
        if potentialFunction=='single-well':
            distRegTerm = 4*Del2(phi)-curvature # compute distance regularization term in equation (13) with the single-well potential p1.
        elif potentialFunction=='double-well':
            distRegTerm=distReg_p2(phi)  #compute the distance regularization term in eqaution (13) with the double-well potential p2.
        else:
            print('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
           
        diracPhi=Dirac(phi,epsilon)
        areaTerm= diracPhi*DistanceMap
        edgeTerm=diracPhi*(vx*Nx+vy*Ny) + diracPhi*g*curvature
        curvatureTerm = diracPhi*curvature
        phi = phi + timestep*(mu*distRegTerm + (curvature>=0)*(lambda_p*edgeTerm)+alfa*areaTerm+(curvature<0)*v*curvatureTerm)
    return phi



#=========================step1: Set parameters===============================
model_path = './checkpoints/'
dataset_name = 'MICCAI09'
#=========================step2: Load Dat====a================================
model_file_name= model_path + "model_tfrecord_279.ckpt"
option_resume = True # load model, resume from previous checkpoint?
data_shape = [256,256,1]
save_path = './output/CGLS/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

#============================================================================
batch_size = 1
mean_file_name = './data/train_1_1.tfrecords_mean.npy'
is_train = False
reuse= True
#=========================step3: Creat network===============================
# Define the batchsize at the begin, you can give the batchsize in x and y_

x = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 1])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1], 2])
fw = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1]])
network,op_class = model_VGG16_Landmark_Detection(x,y_,fw, batch_size,data_shape,reuse=reuse,mean_file_name=mean_file_name, is_train = is_train)
sess = tf.Session()
init=tf.global_variables_initializer()  
sess.run(init)
print("Load existing model " + "!"*10)
if option_resume:
    print("Load existing model initial " + "!"*10)
    saver = tf.train.Saver(network.all_params)
    saver.restore(sess, model_file_name)
 
    
#=========================step4: Find image subpaths===============================
if dataset_name == 'MICCAI09':
    image_paths = []
    data_path = '../MICCAI09/'
    data_subpaths = glob.glob("%s*DICOM*" %(data_path))
    for data_subpath in data_subpaths:
        #============creat path==============
        Temp_path = data_subpath.split(dataset_name)[1]
        save_subpath = save_path+'\\'+Temp_path
        if not os.path.exists(save_subpath):
            os.mkdir(save_subpath)
        #============creat path==============   
        temp_path = glob.glob("%s/*DICOM*" %(data_subpath))[0]
        Temp_path = temp_path.split(dataset_name)[1]
        save_subpath = save_path+'\\'+Temp_path
        if not os.path.exists(save_subpath):
            os.mkdir(save_subpath)
        #============creat path==============   
        temp_path = glob.glob("%s/*DICOM*" %(temp_path))[0]
        Temp_path = temp_path.split(dataset_name)[1]
        save_subpath = save_path+'\\'+Temp_path
        if not os.path.exists(save_subpath):
            os.mkdir(save_subpath)
        #============creat path==============    
        image_subpaths = glob.glob("%s/SC-*" %(temp_path))
        for path in image_subpaths:
            Temp_path = path.split(dataset_name)[1]
            save_subpath = save_path+'\\'+Temp_path
            if not os.path.exists(save_subpath):
                os.mkdir(save_subpath)
            
        image_paths = image_paths + image_subpaths    
    
#=================================================================================
timestep = 1  # time step
mu=0.2  # coefficient of the distance regularization term R(phi)
iter_inner= 4
iter_outer= 5
lambda_p = 30 # coefficient of the weighted length term L(phi)
alfa = 1# coefficient of the weighted area term A(phi)
v = 15
epsilon=1 # papramater that specifies the width of the DiracDelta function
sigma = 1.5 # scale parameter in Gaussian kernel

potentialFunction = 'single-well'
c0=5
thres = 0.7
gama = 0.1
eta = 20
dilate_rate = 3

G=gaussian_kernel_2d_opencv(7,sigma) 
G_size = np.shape(G)
#=================================================================================
Num_img = 0
start_time = time.time()

plt.ion()
fig, axes = plt.subplots(ncols=1)

for image_subpath in image_paths:
    Imagelist = glob.glob("%s/DICOM/*.dcm" %(image_subpath))
    temp_path = image_subpath.split(dataset_name)[1]
    save_subpath = save_path+'\\'+temp_path
    
    temimg = pydicom.read_file(Imagelist[0])
    temimg_shape = temimg.pixel_array.shape
    Phi3D = np.zeros([temimg_shape[0],temimg_shape[1],len(Imagelist)],np.float32)
    IDphi = 0
    for imgmaskname in sorted(Imagelist):   
        imgpathname = imgmaskname[:-4]
        imgname = imgmaskname.split('\\')[5]
        imgname = imgname[:-4]
        dcmfile = pydicom.read_file(imgmaskname)
        dcmdata = dcmfile.pixel_array
        img_shape = dcmdata.shape
        img_ori = MaxMin_normalization(dcmdata)
      
        if img_shape[0]!=data_shape[0] or img_shape[1]!=data_shape[1]:
            img_ori = MaxMin_normalization(img_ori)
            img = Image.fromarray(np.uint8(img_ori))
            img = img.resize((data_shape[0],data_shape[1]))
            img = np.float32(np.array(img))
                    
            imgtemp = np.float32(img[np.newaxis,:,:,np.newaxis])
            feed_dict = {x: imgtemp}
                    
            prediction_class_out= sess.run(op_class, feed_dict=feed_dict) 
            prediction_prob = prediction_class_out[0,:,:,1]
                    
            prediction_class_marker = prediction_prob>thres   
            if np.sum(prediction_class_marker)<=0:
                continue
            #================Post-process=====================
            Mask = Detection_Result_PostProcess(prediction_class_marker)    
    
            q = Mask*100+(prediction_prob<thres)*(gama/(1+np.exp(-prediction_prob)))+(prediction_prob>=thres)*(gama/(1+np.exp(-thres)))*(np.exp(eta*(prediction_prob-thres)))       
            initialLSF = -c0*np.ones([data_shape[0],data_shape[1]]) # generate the initial region R0 as two rectangles
            initialLSF= 2*c0*Mask+initialLSF
                    
            u_ini = np.float32(initialLSF)       
            Img_smooth = signal.convolve2d(img,G,mode='same')
            Ix,Iy = np.gradient(Img_smooth)
            f = np.square(Ix)+np.square(Iy)
            g=1/(1+f) # edge indicator function.
#            g = g**(0.6)
                    
            u = u_ini
            #DistanceMap = tf.Variable(distancemap[np.newaxis,:,:,np.newaxis],dtype=tf.float32,name='input_distance')
            for iii in range(iter_outer):
                u = MarkGuided_LevelSet(u, g, q, lambda_p, mu, alfa, v, epsilon, timestep, iter_inner, potentialFunction)
    
            u= Image.fromarray(np.float32(u))
            u= u.resize((img_shape[1],img_shape[0]))
            u= np.array(u)
            
            Phi3D[:,:,IDphi] = u
            IDphi = IDphi + 1
            show_curve_and_phi(fig,img_ori,u,'r')
            fig.savefig(save_subpath +'/'+imgname+ '_contour_result.png', format='png', transparent=True, dpi=300, pad_inches = 0)
            save_contours(u,imgname,save_subpath)
            
        else:
            img = np.float32(img_ori[np.newaxis,:,:,np.newaxis])
            feed_dict = {x: img}
            prediction_class_out= sess.run(op_class, feed_dict=feed_dict) 
            prediction_prob = prediction_class_out[0,:,:,1]
            sio.savemat(save_subpath +'/'+imgname+'_markprob.mat',{'markprob':prediction_prob})
            prediction_marker = prediction_prob>thres
            if np.sum(prediction_marker)<=0:
                continue
            #================remain the maxmum region=====================
            Mask = Detection_Result_PostProcess(prediction_marker)
    
            q = Mask*100+(prediction_prob<thres)*(gama/(1+np.exp(-prediction_prob)))+(prediction_prob>=thres)*(gama/(1+np.exp(-thres)))*(np.exp(eta*(prediction_prob-thres)))  
            initialLSF = -c0*np.ones(np.shape(img_ori)) # generate the initial region R0 as two rectangles
            initialLSF= 2*c0*Mask+initialLSF
                    
            u_ini = np.float32(initialLSF)
                    
            Img_smooth = signal.convolve2d(img_ori,G,mode='same')
            Ix,Iy = np.gradient(Img_smooth)
            f = np.square(Ix)+np.square(Iy)
            g=1/(1+f) # edge indicator function.
                    
            u = u_ini
            #DistanceMap = tf.Variable(distancemap[np.newaxis,:,:,np.newaxis],dtype=tf.float32,name='input_distance')
            for iii in range(iter_outer):
                u = MarkGuided_LevelSet(u, g, q, lambda_p, mu, alfa,v, epsilon, timestep, iter_inner, potentialFunction)                                

            show_curve_and_phi(fig,img_ori,u,'r')
            fig.savefig(save_subpath +'/'+imgname+'_contour_result.png', format='png', transparent=True, dpi=300, pad_inches = 0)
            save_contours(u,imgname,save_subpath)
            Phi3D[:,:,IDphi] = u
            IDphi = IDphi + 1
            
        
        Num_img = Num_img + 1
    sio.savemat(save_subpath +'/Phi3D.mat',{'Phi3D':Phi3D})  
    print(image_subpath+' is done!\n')
#=========================step4: Train network===============================
sess.close()
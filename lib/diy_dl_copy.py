# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 23:03:04 2019

@author: bettmensch
"""

# contains the dependencies, utils and main classes necessary to use diy deep learning library
#
# Utils/dependencies
# [0] basic imports
# [1] define some util functions
#
# Main layer classes
# [2] define a fully connected layer
# [3.1] define convolution layer utils
# [3.2] define convolution layer main class
# [4] define pooling layer main class
# [5] define fully connected to convolution reshaping layer class
# [6] define convolution to fully connected reshaping layer class
# [7] define dropout layer class
#
# Hidden input layer class
# [8] define 'secret' input layer class
#
# Network class
# [9] define feed forward network class
#
# Genetic Algorithms
# [10] define a generic genetic algorithm
# [11] define gene <-> network weight translator

#----------------------------------------------------
# [0] Make some basic imports
#----------------------------------------------------

import numpy as np
import pandas as pd
from functools import partial
import os
import pickle

#----------------------------------------------------
# [1] Define some computational util functions
#----------------------------------------------------

def tanh(Z):
    return np.tanh(Z)

def Dtanh(A):
    return 1 - np.multiply(A,A)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def Dsigmoid(A):
    return np.multiply(A,1 - A)

def relu(Z,leak=0):
    assert (0 <= leak) and (leak < 0.5)
    return np.max(leak * Z,Z)

def Drelu(A,leak=0):
    t1 = (A > 0) * 1
    t2 = (A <= 0) * leak
    return t1 + t2

def identity(Z):
    return Z

def Didentity(A):
    return np.ones(A.shape)

def softmax(Z):
    #print('From within softmax:')
    #print('Type of input array Z:',type(Z))
    return np.exp(Z) / np.sum(np.exp(Z),axis=1,keepdims=True)

def softmaxLoss(P,Y):
    return -np.mean(np.sum(np.multiply(Y,np.log(P)),axis=1))

def sigmoidLoss(P,Y):
    return -np.mean(np.sum(np.multiply(Y,np.log(P)) + np.multiply((1 - Y),np.log(1 - P)),axis=1))

def l2Loss(P,Y):
    return 0.5 * (np.linalg.norm(P - Y,ord=2) ** 2) / P.shape[0]

def getOneHotY(model,y):
    '''One hot vectorizes a target class index list into a [nData,nClasses] array.'''
                
    # if first time training on classification data with one-hot enabled, remember ordering of class labels used for training
    if str(type(model.classes_ordered)) == "<class 'NoneType'>":
        model.classes_ordered = np.unique(y).reshape(-1)
            
    # convert class labels into indices as specified in classes_ordered attribute
    y_numeric = np.array([np.where(model.classes_ordered == y_i)[0][0] for y_i in y]).reshape(-1)
        
    # create one-hot encoded target matrix from class labels y and class label <-> class index assignment rule stored in classes_ordered attribute
    nClasses = len(model.classes_ordered)
    Y = np.eye(nClasses)[y_numeric]
        
    return Y

def getBatches(X,Y,batchSize):
    '''Sample randomly from X and Y, then yield batches.'''
    nData = X.shape[0]
    shuffledIndices = np.arange(nData)
    np.random.shuffle(shuffledIndices)
    
    XShuffled, YShuffled = X[shuffledIndices], Y[shuffledIndices]
    
    nBatches = int(X.shape[0] / batchSize)
    
    for iBatch in range(nBatches):
        XBatch, YBatch = (XShuffled[iBatch*batchSize:(iBatch+1)*batchSize],
                          YShuffled[iBatch*batchSize:(iBatch+1)*batchSize])
    
        yield XBatch, YBatch, nBatches
        
def save_model(model,save_dir,model_name,verbose = True):
    '''Helper function to save trained models for later use.'''
    
    # if directory doesnt exist already, create it
    if (not os.path.isdir(save_dir)):
        if verbose:
            print("Directory " + str(save_dir) + " doesnt exist yet and will be created.")
            
        os.mkdir(save_dir)
        
    # create save path with specified name
    full_save_path = os.path.join(save_dir,model_name)
    
    # save model as pickled object file
    if verbose:
        print("Saving model object in " + str(full_save_path))
    
    with open(full_save_path,'wb') as saved_model_file:
        pickle.dump(model,saved_model_file)
        
    if verbose:
        print("Finished saving model object in " + str(full_save_path))
        
    return full_save_path
    
class SGD(object):
    '''Class representing the stochastic gradient descent with momentum algorithm'''
    
    def __init__(self,eta,gamma,epsilon,lamda,batchSize):
        '''eta: learning rate parameter (goes with cost function gradient)
        gamma: momentum coefficient (goes with previous parameter update)
        lamda: regularization parameter
        batchSize: batch size used during network weight optimization'''
        
        # store relevant optimization hyper-parameters
        self.eta = eta
        self.gamma = gamma
        self.lamda = lamda
        self.batchSize = batchSize
        
        # initialize storage for previous updates
        self.DeltaWeightPrevious = 0
        self.DeltaBiasPrevious = 0
        
    def get_parameter_updates(self,
                              current_weight,
                              current_bias,
                              Dcache):
        '''Takes a gradient cache in dictionary form and computes and stores
        layer parameters updates. Stores the current updates for next iteration.'''
        
        # get weight update
        DeltaWeight = - self.eta * Dcache['DWeight'] \
                        - self.gamma * self.DeltaWeightPrevious \
                        - self.lamda * current_weight / self.batchSize
        
        # get bias update
        DeltaBias = - self.eta * Dcache['Dbias'] \
                        - self.gamma * self.DeltaBiasPrevious

        # store updates for next iteration's momentum contribution
        self.DeltaWeightPrevious = DeltaWeight
        self.DeltaBiasPrevious = DeltaBias
        
        return DeltaWeight, DeltaBias
    
    def __str__(self):
        '''Returns a string with bio of SGD optimizer.'''

        bio = '\t \t Optimization parameters--------------------------------' \
                + '\n Learning rate:' + str(self.eta) \
                + '\n Regularization parameter used:' + str(self.lamda) \
                + '\n Batch size used: ' + str(self.batchSize) \
                + '\n Type of optimization used: SGD'
            
        return bio
    
#----------------------------------------------------
# [2] Fully connected layer class
#----------------------------------------------------

class FcLayer(object):
    '''Object class representing a fully connected layer in a feed-forward neural network'''
    def __init__(self,n,activation='tanh',leak=0):
        self.sizeIn = None
        self.sizeOut = [n]
        
        assert activation in ['tanh','sigmoid','relu','identity','softmax']
        if activation == 'tanh':
            self.activation = (tanh,activation)
            self.Dactivation = Dtanh
        elif activation == 'sigmoid': # possibly output layer, attach loss function just in case
            self.activation = (sigmoid,activation)
            self.Dactivation = Dsigmoid
        elif activation == 'relu':
            self.leak = leak
            self.activation = (relu,activation)
            self.Dactivation = Drelu
        elif activation == 'identity':
            self.activation = (identity,activation)
            self.Dactivation = Didentity
        elif activation == 'softmax': # definitely output layer, so no need for Dactivation
            self.activation = (softmax,activation)
        
        # initialize layer parameters and activation cache
        self.Weight = None
        self.bias = None
        self.cache = {'A':None,'DZ':None}
        
        # initialize neighbouring layer contact points, i.e 'buy address book'
        self.previousLayer = None
        self.nextLayer = None
        
        # set up optimization configs
        self.optimizer = None
        self.has_optimizable_params = True
        
    def __str__(self):
        '''Returns a string with bio of layer.'''

        bio = '----------------------------------------' \
                + '\n Layer type: Fully connected layer' \
                + '\n Number of neurons in input data: ' + str(self.sizeIn) \
                + '\n Type of activation used: ' + self.activation[1] \
                + '\n Number of neurons in output data: ' + str(self.sizeOut) \
            
        return bio
        
    def forwardProp(self):
        # retrieve previous layer's activation
        A_p = self.previousLayer.cache['A']
        
        # calculate this layer's pre-activation
        Z_c = np.dot(A_p,self.Weight) + self.bias
        
        # calculate this layer's activation
        # if using relu activation, use layer's leak attribute
        if self.activation[1]=='relu':
            A_c = self.activation[0](Z_c,self.leak)
        # if not using relu activation, no hyperparameters are needed in this step
        else:
            A_c = self.activation[0](Z_c)
        #print("Type of A_c:", type(A_c))
        #print("From within fully connected layer's forwardProp:")
        #print("Shape of previous layer's activation:",A_p.shape)
        #print("Shape of current layer's activation:",A_c.shape)
        #print("------------------------------------")
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        A_c = self.cache['A']
        if self.activation[1]=='relu':
            self.cache['DZ'] = np.multiply(self.Dactivation(A_c,self.leak),DA_c)
        else:
            self.cache['DZ'] = np.multiply(self.Dactivation(A_c),DA_c)
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        A_p = self.previousLayer.cache['A']
        
        # calculate DZ_p, i.e. DZ of previous layer in network
        DA_p = np.dot(DZ_c,self.Weight.T)
        self.previousLayer.getDZ_c(DA_p)
        
        # calculate weight gradients
        DWeight = np.dot(A_p.T,DZ_c) / A_p.shape[0]
        Dbias = np.mean(DZ_c,axis=0)
        Dcache = {'DWeight':DWeight,'Dbias':Dbias}#, 
                  #'Weight':self.Weight, 'bias':self.bias}
        
        return Dcache
        
    def updateLayerParams(self,Dcache,direction_coeff = 1):
        
        # retrieve updates for layer's weights and bias using layer's optimizer
        DeltaWeight, DeltaBias = self.optimizer.get_parameter_updates(self.Weight,
                                                                      self.bias,
                                                                      Dcache)
        
        # update parameters with respective updates obtained from optimizer
        self.Weight += direction_coeff * DeltaWeight
        self.bias += direction_coeff * DeltaBias
        
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer
            
        self.initializeWeightBias()
        
    def initializeWeightBias(self):
        n_p,n_c = self.sizeIn[0],self.sizeOut[0]
        self.Weight = np.random.randn(n_p,n_c) * 1 / (n_p + n_c)
        self.bias = np.zeros((1,n_c))

#----------------------------------------------------
# # [3.1] define convolution layer utils
#----------------------------------------------------

def getPictureDims(height_pl,width_pl,padding_cl,kernelParams_cl):
    '''Calculates a convolutional layers height and width dimensions based on:
    - previous (convolutional) layer shape
    - type of padding used
    - kernel size of curent layer'''
    
    stride = kernelParams_cl['stride']
    if 'height_k' in kernelParams_cl: # dealing with a convolutional layer's request
        height_k, width_k = kernelParams_cl['height_k'],kernelParams_cl['width_k']
    elif 'height_pool' in kernelParams_cl: # dealing with a pooling layer's request
        height_k, width_k = kernelParams_cl['height_pool'],kernelParams_cl['width_pool']
    
    if padding_cl == 'valid':
        height_pad, width_pad = (0,0)
    if padding_cl == 'same':
        height_pad = np.ceil((stride*(height_pl-1)+height_k-height_pl) / 2)
        width_pad = np.ceil((stride*(width_pl-1)+width_k-width_pl) / 2)
        
    height_cl = int(np.ceil((height_pl-height_k+1+2*height_pad) / stride))
    width_cl = int(np.ceil((width_pl-width_k+1+2*width_pad) / stride))
    
    return height_cl, width_cl

def getConvSliceCorners(h,w,height_k,width_k,stride):
    '''Calculates and returns the edge indices of the slice in layer_p used to compute layer_c[h,w]'''
    hStart, hEnd = (h * stride, h * stride + height_k)
    wStart, wEnd = (w * stride, w * stride + width_k)
    
    return hStart,hEnd,wStart,wEnd

def pad(Z,pad):
    # Takes a four-dimensionan tensor tensor of shape (x1,x2,x3,x4,x5)
    # Adds zero padding for dimensions x2 and x3 to create an array
    # Zpadded of shape (x1,x2+2*pad,x3*pad,x4)
    Zpadded = np.pad(Z,mode='constant',
                     pad_width=((0,0),(0,0),(pad,pad),(pad,pad),(0,0)),
                    constant_values=((0,0),(0,0),(0,0),(0,0),(0,0)))
    
    return Zpadded

#----------------------------------------------------
# [3.2] define convolution layer main class
#----------------------------------------------------

class ConvLayer(object):
    '''Object class representing a convolutional layer in a feed-forward neural net'''
    
    def __init__(self,kernelHeight,kernelWidth,channels,
                 stride,padding='valid',
                 activation='tanh',
                 leak=0):
    
        assert padding in ['same','valid']
        assert activation in ['tanh','sigmoid','relu','identity']
        
        if activation == 'tanh':
            self.activation = (tanh,activation)
            self.Dactivation = Dtanh
        elif activation == 'sigmoid':
            self.activation = (sigmoid,activation)
            self.Dactivation = Dsigmoid
        elif activation == 'relu':
            self.leak = leak
            self.activation = (relu,activation)
            self.Dactivation = Drelu
        elif activation == 'identity':
            self.activation = (identity,activation)
            self.Dactivation = Didentity
            
        self.padding = padding
        self.sizeIn = None
        self.sizeOut = [channels]
        self.kernelParams = {'stride':stride,'height_k':kernelHeight,'width_k':kernelWidth}
        
        self.Weight = None
        self.bias = None
        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
        # set up optimization configs
        self.optimizer = None
        self.has_optimizable_params = True
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        
        bio = '----------------------------------------' \
                + '\n Layer type: Convolution layer' \
                + '\n Shape of kernel (width,height): ' + ','.join([str(self.kernelParams['height_k']),
                                                                  str(self.kernelParams['width_k'])]) \
                + '\n Stride used for kernel: ' + str(self.kernelParams['stride']) \
                + '\n Shape of input data (channels,height,width): ' + str(self.sizeIn) \
                + '\n Padding used: ' + self.padding \
                + '\n Type of activation used: ' + self.activation[1] \
                + '\n Shape of output data (channels,height,width): ' + str(self.sizeOut)
            
        return bio

        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_k = self.kernelParams['height_k']
        width_k = self.kernelParams['width_k']
        stride = self.kernelParams['stride']
        
        Z_c = np.zeros((batchSize,channels_c,height_c,width_c,1))
        
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_k,width_k,stride)
                X_hw = np.multiply(A_p[:,:,hStart:hEnd,wStart:wEnd,:],self.Weight)
                Y_hw = np.sum(X_hw,axis=(1,2,3))
                Z_c[:,:,h,w,0] = Y_hw
        
        Z_c += self.bias
        
        # calculate this layer's activation in case of relu activaions
        if self.activation[1]=='relu':
            A_c = self.activation[0](Z_c,self.leak)
        # calcualte this layer's activation for non-relu activation types
        else:
            A_c = self.activation[0](Z_c)
            
        # store this layer's activation in cache
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        A_c = self.cache['A']
        
        # calculate DL/DZ for current layer in case of relu activation
        if self.activation[1]=='relu':
            self.cache['DZ'] = self.Dactivation(A_c,self.leak) * DA_c
        # calculate DL/DZ for current layer in case of-non relu activation types
        else:
            self.cache['DZ'] = self.Dactivation(A_c) * DA_c
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        # get curent layer's and previous layer's architectural parameters
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_k = self.kernelParams['height_k']
        width_k = self.kernelParams['width_k']
        stride = self.kernelParams['stride']
        
        channels_p = self.sizeIn[0]
        height_p = self.sizeIn[1]
        width_p = self.sizeIn[2]
        
        # calculate weight gradients & DZ_p, i.e. DZ of previous layer in network
        DWeight = np.zeros((1,channels_p,height_k,width_k,channels_c))
        DZ_cback = np.transpose(DZ_c,(0,4,2,3,1))
        
        DA_p = np.zeros((batchSize,height_p,width_p,channels_p,1))
        Weight_back = np.transpose(self.Weight,(0,4,2,3,1))
        
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_k,width_k,stride)
                #print('A')
                I_hw = np.multiply(DZ_cback[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:],
                                   A_p[:,:,hStart:hEnd,wStart:wEnd,:])

                J_hw = np.mean(I_hw,axis=0)

                DWeight[0,:,:,:,:] += J_hw
                
                X_hw = np.multiply(DZ_c[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:],
                                   Weight_back)
                #print('E')
                Y_hw = np.sum(X_hw,axis=1)

                DA_p[:,hStart:hEnd,wStart:wEnd,:,0] += Y_hw

        # make previous layer produce its own DL/DZ
        DA_p = np.transpose(DA_p,(0,3,1,2,4))
        self.previousLayer.getDZ_c(DA_p)
        
        # for this layer, calculate DL/Db
        Dbias = np.mean(np.sum(DZ_c,axis=(2,3,4)),axis=0)
        Dbias = Dbias[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        
        # for this layer, store derivatives and weights in cache for the optimizer
        Dcache = {'DWeight':DWeight,'Dbias':Dbias}#, 
                  #'Weight':self.Weight, 'bias':self.bias}
        
        return Dcache
    
    def updateLayerParams(self,Dcache,direction_coeff=1):
        # retrieve updates for layer's weights and bias using layer's optimizer
        DeltaWeight, DeltaBias = self.optimizer.get_parameter_updates(self.Weight,
                                                                      self.bias,
                                                                      Dcache)
        
        # update parameters with respective updates obtained from optimizer
        self.Weight += direction_coeff * DeltaWeight
        self.bias += direction_coeff * DeltaBias
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        height_pl, width_pl = self.sizeIn[1],self.sizeIn[2]
        height_cl, width_cl = getPictureDims(height_pl,
                                           width_pl,
                                           self.padding,
                                           self.kernelParams)
        
        self.sizeOut.extend([height_cl,width_cl])
        
        self.nextLayer = nextLayer
            
        self.initializeWeightBias()
        
    def initializeWeightBias(self):
        self.Weight = np.random.randn(1,
                                      self.sizeIn[0], # channels_pl
                                      self.kernelParams['height_k'], # kernel height
                                      self.kernelParams['width_k'], # kernel width
                                      self.sizeOut[0]) # channels_cl
        self.bias = np.zeros((1,self.sizeOut[0],1,1,1)) # channels_cl

#----------------------------------------------------
# [4] define pooling layer main class
#----------------------------------------------------

class PoolingLayer(object):
    '''Pooling layer (either "max" or "mean") between convolutional/appropriate reshaping layers'''
    
    def __init__(self,poolingHeight,poolingWidth,
                 stride,padding='valid',
                 poolingType='max'):
        
        assert padding in ['same','valid']
        assert poolingType in ['max','mean']
        
        self.padding = padding
        self.poolingType = poolingType
        self.sizeIn = None
        self.sizeOut = None
        self.poolingParams = {'stride':stride,'height_pool':poolingHeight,'width_pool':poolingWidth}
        
        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
        # set up optimization configs
        self.has_optimizable_params = False
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        
        bio = '----------------------------------------' \
                + '\n Layer type: Pooling layer (' + str(self.poolingType) +')' \
                + '\n Shape of pool (width,height): ' + ','.join([str(self.poolingParams['height_pool']),
                                                                  str(self.poolingParams['width_pool'])]) \
                + '\n Stride used for pool: ' + str(self.poolingParams['stride']) \
                + '\n Shape of input data (channels,height,width): ' + str(self.sizeIn) \
                + '\n Padding used: ' + self.padding \
                + '\n Shape of output data (channels,height,width): ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_pool = self.poolingParams['height_pool']
        width_pool = self.poolingParams['width_pool']
        stride = self.poolingParams['stride']
        
        Z_c = np.zeros((batchSize,channels_c,height_c,width_c,1))
                
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_pool,width_pool,stride)
                X_hw = A_p[:,:,hStart:hEnd,wStart:wEnd,0]
                
                if self.poolingType == 'max':
                    Y_hw = np.amax(X_hw,axis=(2,3))
                elif self.poolingType == 'mean':
                    Y_hw = np.mean(X_hw,axis=(2,3))
                    
                Z_c[:,:,h,w,0] = Y_hw
        
        self.cache['A'] = Z_c # pooling layer has no activation, hence A_c = Z_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        self.cache['DZ'] = DA_c # since for pooling layers A = Z -> DA = DZ
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        
        channels_c = self.sizeOut[0]
        height_c = self.sizeOut[1]
        width_c = self.sizeOut[2]
        
        height_pool = self.poolingParams['height_pool']
        width_pool = self.poolingParams['width_pool']
        stride = self.poolingParams['stride']
        
        channels_p = self.sizeIn[0]
        height_p = self.sizeIn[1]
        width_p = self.sizeIn[2]

        # calculate weight gradients & DZ_p, i.e. DZ of previous layer in network
        
        DA_p = np.zeros((batchSize,channels_p,height_p,width_p,1))
        
        for h in range(height_c):
            for w in range(width_c):
                hStart,hEnd,wStart,wEnd = getConvSliceCorners(h,w,height_pool,width_pool,stride)
                
                if self.poolingType == 'max':
                    X_hw = A_p[:,:,hStart:hEnd,wStart:wEnd,0]
                    #print("From within pooling layer's backprop:")
                    #print("Shape of X_hw:",X_hw.shape)
                    Y_hw = np.amax(X_hw,(2,3))[:,:,np.newaxis,np.newaxis]
                    #print("Shape of Y_hw:",Y_hw.shape)
                    U_hw = DZ_c[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:]
                    #print('Shape of U_hw:',U_hw.shape)
                    V_hw = (X_hw==Y_hw)[:,:,:,:,np.newaxis]
                    #print('Shape of V_hw:',V_hw.shape)
                    #print('Shape of updated DA_p slice:',DA_p[:,:,hStart:hEnd,wStart:wEnd,:].shape)
                    DA_p[:,:,hStart:hEnd,wStart:wEnd,:] += U_hw * V_hw
                elif self.poolingType == 'mean':
                    X_hw = 1/(height_pool * width_pool) * np.ones(batchSize,channels_p,height_pool,width_pool,1)
                    DA_p[:,:,hStart:hEnd,wStart:wEnd,:] += np.multiply(DZ_c[:,:,h,w,:][:,:,np.newaxis,np.newaxis,:],
                                                                       X_hw)
        
        self.previousLayer.getDZ_c(DA_p)
        
        return
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.sizeOut = [self.sizeIn[0]] # number of channels remains unchanged through pooling
        height_pl, width_pl = self.sizeIn[1],self.sizeIn[2]
        height_cl, width_cl = getPictureDims(height_pl,
                                           width_pl,
                                           self.padding,
                                           self.poolingParams)
        
        self.sizeOut.extend([height_cl,width_cl])
        
        self.nextLayer = nextLayer

#----------------------------------------------------
# [5] define fully connected to convolution reshaping layer class
#----------------------------------------------------

class FcToConv(object):
    '''Transitional layer handling reshaping between fclayer activations -> convLayer activations,
    and convLayer activation derivatives -> fcLayer activation derivatives.'''
    
    def __init__(self,convDims):

        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
        self.sizeIn = None
        
        [convChannels,convHeight,convWidth] = convDims
        self.sizeOut = [convChannels,convHeight,convWidth]
        
        # set up optimization configs
        self.has_optimizable_params = False
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        bio = '----------------------------------------' \
                + '\n Layer type: Reshaping layer (Fully connected -> Convolution)' \
                + '\n Shape of input data: ' + str(self.sizeIn) \
                + '\n Shape of output data: ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        aShape = [batchSize,self.sizeOut[0],self.sizeOut[1],self.sizeOut[2],1]
        A_c = Z_c = A_p.reshape(aShape)
        #print("From within reshape (conv -> fc) layer's forwardProp:")
        #print("Shape of previous layer's activation:",A_p.shape)
        #print("Shape of current layer's activation:",A_c.shape)
        #print("------------------------------------")
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        batchSize = DA_c.shape[0]
        dzShape = [batchSize, self.sizeIn[0]]
        self.cache['DZ'] = DA_c.reshape(dzShape)
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        # calculate DZ_p, i.e. DZ of previous layer in network
        DA_p = DZ_c
        self.previousLayer.getDZ_c(DA_p)
        
        return
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer
        self.convShape = self.nextLayer.convShape

#----------------------------------------------------
# [6] define convolution to fully connected reshaping layer class
#----------------------------------------------------

class ConvToFC(object):
    '''Transitional layer handling reshaping between convlayer activations -> fcLayer activations,
    and fcLayer activation derivatives -> convLayer activation derivatives.'''
    
    def __init__(self,n):

        self.cache = {'A':None,'DZ':None}
        self.previousLayer = None
        self.nextLayer = None
        
        self.sizeIn = None
        self.sizeOut = [n]
        
        # set up optimization configs
        self.has_optimizable_params = False
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        bio = '----------------------------------------' \
                + '\n Layer type: Reshaping layer (Convolution -> Fully connected)' \
                + '\n Shape of input data: ' + str(self.sizeIn) \
                + '\n Shape of output data: ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        batchSize = A_p.shape[0]
        #A_c = A_p.reshape([batchSize].extend(self.sizeOut))
        A_c = A_p.reshape([batchSize,self.sizeOut[0]])
        #print("From within reshape (conv -> fc) layer's forwardProp:")
        #print("Shape of previous layer's activation:",A_p.shape)
        #print("Shape of current layer's activation:",A_c.shape)
        #print("------------------------------------")
        self.cache['A'] = A_c
        
    def getDZ_c(self,DA_c):
        # calculates this layer's DZ. gets called from next layer in network during backprop
        batchSize = DA_c.shape[0]
        #dzShape = [batchSize].extend(self.sizeIn).extend(1)
        dzShape = [batchSize,self.sizeIn[0],self.sizeIn[1],self.sizeIn[2],1]
        self.cache['DZ'] = DA_c.reshape(dzShape)
        
    def backwardProp(self):
        # get stored activation values
        DZ_c = self.cache['DZ']
        # calculate DZ_p, i.e. DZ of previous layer in network
        DA_p = DZ_c
        self.previousLayer.getDZ_c(DA_p)
        
        return
    
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer

#----------------------------------------------------
# [7] define dropout layer class
#----------------------------------------------------

class Dropout(object):
    '''Dropout layer acting between "real" network layers.'''
    
    def __init__(self,dropoutRate):
        # set layer specific (non-optimizable) parameter
        self.dropoutRate = dropoutRate
        
        # initalize activation cache
        self.cache = {'A':None,'DZ':None,'outDropper':None}
        
        # initialize neighbouring layer contact points, i.e. 'buy address book'
        self.previousLayer = None
        self.nextLayer = None
        
        # setup optimization config
        self.has_optimizable_params = False
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        bio = '----------------------------------------' \
                + '\n Layer type: Dropout layer' \
                + '\n Dropout rate: ' + str(self.dropoutRate) \
                + '\n Shape of input/output data: ' + str(self.sizeOut)
            
        return bio
        
    def forwardProp(self):
        A_p = self.previousLayer.cache['A']
        outDropper = np.random.choice([0,1],size=A_p.shape,p=[self.dropoutRate,1-self.dropoutRate])
        A_c = Z_c = np.multiply(outDropper,A_p)
        self.cache['A'] = A_c
        self.cache['outDropper'] = outDropper
        
    def getDZ_c(self,DA_c):
        DZ_c = DA_c
        self.cache['DZ'] = DZ_c
        
    def backwardProp(self):
        DZ_c = self.cache['DZ']
        outDropper = self.cache['outDropper']
        DA_p = np.multiply(DZ_c,outDropper)
        self.previousLayer.getDZ_c(DA_p)
        
        return
        
    def makeReady(self,previousLayer=None,nextLayer=None):
        self.previousLayer = previousLayer
        self.sizeIn = self.sizeOut = self.previousLayer.sizeOut
        
        self.nextLayer = nextLayer

#----------------------------------------------------
# [8] define 'secret' input layer
#----------------------------------------------------

class _InputLayer(object):
    '''Input layer created automatically by network once the training data shape/kind is known to it.'''
    
    def __init__(self,pad=0):
        self.sizeOut = None
        self.cache = {'A':None}
        self.nextLayer = None
        self.flatData = None
        self.pad = pad
        
    def __str__(self):
        '''Returns a string with bio of layer.'''
        
        if self.flatData:
            secondLine = '\n Number of neurons in input/output data: ' + str(self.sizeOut)
        elif not self.flatData:
            secondLine = '\n Shape of input/output data (channels,height,width): ' + str(self.sizeOut)
        
        bio = '----------------------------------------' \
                + '\n Layer type: (Secret) input layer' \
                + secondLine
            
        return bio
        
    def forwardProp(self,XBatch):
        if self.flatData:
            self.cache['A'] = XBatch # -> input data is 2 dimensional, (bachSize,nFeature)
            #print("From within secret input layer's forward prop:")
            #print("Type of input batch without adding bogus dimension:",type(self.cache['A']))
        elif not self.flatData:
            self.cache['A'] = pad(np.expand_dims(XBatch,-1),self.pad) # -> input data is 4 dim.,(sampleSize,channels,height,width)
            #print("From within secret input layer's forward prop:")
            #print("Shape of input batch after adding bogus dimension:",self.cache['A'].shape)
            
        #print("-------------------------------")
        
        return
    
    def getDZ_c(self,DA_c):
        # bogus function for layer consistency from the neural net class point of view
        return
    
    def makeReady(self,nextLayer=None,XSample=None):
        self.nextLayer = nextLayer
        self.sizeOut = self.getSizeOutFromX(XSample)
        
    def getSizeOutFromX(self,XSample):
        if len(XSample.shape) == 2: # -> assume X is flattened array of shape (sampleSize,nFeature)
            sizeOut = [XSample.shape[1]]
            
            self.flatData = True
            
            return sizeOut
        
        elif len(XSample.shape) == 4: # -> assume X is high dim. tensor of shape (sampleSize,channels,height,width)
            inputChannels,inputHeight,inputWidth = XSample.shape[1:]
            sizeOut = [inputChannels,inputHeight+2*self.pad,inputWidth+2*self.pad]
            
            self.flatData = False
            
            return sizeOut
        
        else:
            print('''X has to be either of shape [nSamples,nFeatures] or, for images,
            [imageChannels,imageHeight,imageWidth]. Please reshape your training data and
            try compiling the model again.''')

#----------------------------------------------------
# [9] define feed forward network class
#----------------------------------------------------

class FFNetwork(object):
    
    def __init__(self,initPad=0):
        self.initPad = initPad
        self.layers = []
        self.loss = None
        self._inputLayer = None
        self.dataType = None # will indicate wether flattened feature vectors or high dim image tensors
        self.finalState = False
        self.trained = False
        self.classes_ordered = None
        
    def __str__(self):
        '''Print out structure of neural net (if it has been fixated).'''
        
        if self.finalState:
            bluePrint = '\n'.join([self._inputLayer.__str__()] + [layer.__str__() for layer in self.layers])
            
            return bluePrint
        else:
            print('The model has to be fixated first.')
        
    def addFCLayer(self,n,activation='tanh'):
        '''Adds a fully connected layer to the neural network.'''
        
        if not self.finalState:
            # if network has not been fixated yet, create fully connected layer
            # and add to network
            fullyConnectedLayer = FcLayer(n,activation)
            self.layers.append(fullyConnectedLayer)
        else:
            print('The network has already been fixated. No further layers can be added.')
        
    def addConvLayer(self,kernelHeight,kernelWidth,channels,stride,padding='valid',activation='tanh'):
        '''Adds a convolution layer to the neural network.'''
        
        if not self.finalState:
            # if network has not been fixated yet, create convolution layer
            # and add to network
            convolutionLayer = ConvLayer(kernelHeight,
                                         kernelWidth,
                                         channels,
                                         stride,
                                         padding,activation)
            self.layers.append(convolutionLayer)
        else:
            print('The network has already been fixated. No further layers can be added.')
        
    def addPoolingLayer(self,poolingHeight,poolingWidth,stride,padding='valid',poolingType='max'):
        '''Adds a pooling layer to the neural network. Recommended after convolutional layers.'''
        
        if not self.finalState:
            # if network has not been fixated yet, create convolution layer
            # and add to network
            poolingLayer = PoolingLayer(poolingHeight,
                                        poolingWidth,
                                        stride,
                                        padding,
                                        poolingType)
            self.layers.append(poolingLayer)
        else:
            print('The network has already been fixated. No further layers can be added.')

        
    def addFCToConvReshapeLayer(self,convDims):
        '''Adds a reshaping layer to the neural network. Necessary to link up a fully connected layer
        with a subsequent convolution layer.'''
        
        if not self.finalState:
            # if network has not been fixated yet, create convolution layer
            # and add to network
            shapeFullyConnectedToConvolution = FcToConv(convDims)
            self.layers.append(shapeFullyConnectedToConvolution)
        else:
            print('The network has already been fixated. No further layers can be added.')
        
    def addConvToFCReshapeLayer(self,n):
        '''Adds a reshaping layer to the neural network. Necessary to link up a convolutional layer with a 
        subsequent fully connected layer.'''
        
        if not self.finalState:
            # if network has not been fixated yet, create convolution layer
            # and add to network
            shapeConvolutionalToFullyConnected = ConvToFC(n)
            self.layers.append(shapeConvolutionalToFullyConnected)
        else:
            print('The network has already been fixated. No further layers can be added.')
        
    def addDropoutLayer(self,dropoutRate):
        '''Adds a dropout layer.'''
        
        if not self.finalState:
            # if network has not been fixated yet, create convolution layer
            # and add to network
            dropoutLayer = Dropout(dropoutRate)
            self.layers.append(dropoutLayer)
        else:
            print('The network has already been fixated. No further layers can be added.')
        
    def fixateNetwork(self,XSample):
        '''Fixes model, finalising its blue-print.
        Attaches loss function to model.
        Creates hidden input layer based on shape of passed sample.
        Calls each layer's makeReady() method.'''
        
        # only do stuff if model hasnt allready been fixated
        if self.finalState:
            print('This model has already been fixated.')
            
            return
        
        # add secret input layer and make ready
        self._inputLayer = _InputLayer(self.initPad)
        self._inputLayer.makeReady(self.layers[0],XSample)
        
        # iterate through layers and introduce to immediate neighouring layers
        for i, layer in enumerate(self.layers):
            if i == 0: # first layer, use _inputLayer as previous layer
                previousLayer = self._inputLayer
            else: # at least second layer, so pass previous layer
                previousLayer = self.layers[i-1]
            
            if i == len(self.layers) - 1: # last user made layer in network, no next layer exists
                nextLayer = None
            else: # at most second to last layer, pass next layer
                nextLayer = self.layers[i+1]
                
            layer.makeReady(previousLayer,nextLayer)
        
        lastLayer = self.layers[-1]
        
        # attach loss function to neural net depending on last fully connected layer's activation type
        if lastLayer.activation[0] == sigmoid:
            self.loss = sigmoidLoss
        elif lastLayer.activation[0] == softmax:
            self.loss = softmaxLoss
        else:
            print('The last layer needs to have either "softmax" or "sigmoid" activation. Model was not fixated')
        
        self.finalState = True
        
    def trainNetwork(self,
                     X,y,
                     nEpochs=5,
                     batchSize=25,
                     optimizer='sgd',
                     eta=0.001,
                     gamma=0.99,
                     epsilon=0.0000001,
                     lamda=0,
                     displaySteps=50,
                     oneHotY = True):
        '''Trains the neural network using naive gradient descent.'''
        # vectorize Y to one-hot format if needed (default is True)
        if oneHotY:
            Y = self.oneHotY(y)
        elif not oneHotY:
            Y = y
        
        # initialize storage for batch losses to be collected during training
        lossHistory = []
        recentLoss = 0
        
        # create and attach specified optimizers to layers, also keep one for later reference
        self.initialize_layer_optimizers(optimizer,eta,gamma,epsilon,lamda,batchSize)
        
        # execute training
        for epoch in range(nEpochs):
            for i,(XBatch,YBatch,nBatches) in enumerate(getBatches(X,Y,batchSize)):
                P = self.forwardProp(XBatch)
                batchLoss = self.loss(P,YBatch)
                recentLoss += batchLoss
                self.backwardProp(YBatch)
                
                if ((i+1) % displaySteps) == 0:
                    averageRecentLoss = recentLoss / displaySteps
                    lossHistory.append(averageRecentLoss)
                    recentLoss = 0
                    print('Epoch: ',str(epoch+1),'/',str(nEpochs))
                    print('Batch: ',str(i+1),'/',str(nBatches))
                    print('Loss averaged over last ',str(displaySteps),
                          ' batches: ',str(averageRecentLoss))
                    print('---------------------------------------------------')
        
        # announce end of training
        self.trained = True
        print('---------------------------------------------------')
        print('Training finished.')
        print('nEpochs:',str(nEpochs))
        print('Optimizer used:',str(self.most_recent_optimizer_used))
        print('batchSize:',str(batchSize))
        
        self.lossHistory = lossHistory
        
        return 
        
    def oneHotY(self,y):
        '''One hot vectorizes a target class index list into a [nData,nClasses] array.'''
                
        return getOneHotY(self,y)
    
    def initialize_layer_optimizers(self,optimizer,eta,gamma,epsilon,lamda,batchSize):
        '''Creates and attaches optimizer objects to all network layers carrying optimizable parameters'''
        
        # make sure library supports specified optimization type
        assert optimizer in ['sgd']
        
        # get optimizer creator, i.e. 'prep optimizer cloning machine'
        if optimizer == 'sgd':
            optimizer_class = SGD
            
        # attach one optimizer to network for later reference
        self.most_recent_optimizer_used = optimizer_class(eta,gamma,epsilon,lamda,batchSize)
        
        # create and attach optimizer to network layers which have optimizable parameters
        for layer in self.layers:
            # check if layer has optimizable parameters
            if layer.has_optimizable_params:
                layer.optimizer = optimizer_class(eta,gamma,epsilon,lamda,batchSize)
        
    def forwardProp(self,XBatch):
        '''Executes one forward propagation through the network. Returns the loss averaged over the batch.'''
        
        self._inputLayer.forwardProp(XBatch) # feed training batch into network
        
        for layer in self.layers: # forward propagate through the network
            layer.forwardProp()
            
        P = self.layers[-1].cache['A'] # get prediction
            
        return P
        
    def backwardProp(self,YBatch,reinforcement_coeff=1):
        '''Executes one backward propagation through the network. Updates the network's weights.'''
        
        P = self.layers[-1].cache['A']
        self.layers[-1].cache['DZ'] = (P - YBatch) #/ YBatch.shape[0]
        
        for i,layer in enumerate(reversed(self.layers)):
            # propagate loss function gradient backwards through network
            layerDcache = layer.backwardProp()
            if layer.has_optimizable_params:
                # where sensible, update parameters
                layer.updateLayerParams(layerDcache,
                                        direction_coeff=reinforcement_coeff)
            
    def predict(self,X,distribution = False):
        '''If model is trained, performs forward prop and returns the prediction array.'''
        
        if not self.trained:
            print('Model needs to be trained first.')
            
            return
        
        P = self.forwardProp(X)
        
        # if classification model, default mode is to return argmax class array of shape (m,1)
        if (not str(type(self.classes_ordered)) == "<class 'NoneType'>") and (distribution == False):
            # get indices (w.r.t classes_ordered ordering) of classes with max cond. prob.
            class_inds = np.argmax(P,axis=1).reshape(-1)
            # get predicted class labels in column array
            P_class = self.classes_ordered[class_inds].reshape(-1,1)
            
            return P_class
        # if classification model but distribution specifically specified OR regression model, return outputs as is
        else:
            #print("AAA")
            return P
        
    def save(self,save_dir,model_name,verbose = True):
        '''Save model object as pickled file by calling the algorithm generic save function defined in section [1].'''
        
        return save_model(model = self, save_dir = save_dir, model_name = model_name, verbose = verbose)
    
#----------------------------------------------------
# [10] define genetic algorithm
#----------------------------------------------------
        
class GA(object):
    '''Class representing a continuouss genetic algorithm '''
    
    def __init__(self,
                 dna_seq_len,
                 gene_type = 'continuous',
                 initializer = None):
        
        # store DNA sequence length
        self.dna_seq_len = dna_seq_len
        
        # initialize cost function
        self.cost_function = None
        
        # initialize gene history storage data frame
        self.population_history = None
        
        # store type
        self.gene_type = gene_type
        
        # method to initialize first generation of genes
        # needs to be of the form initializer(n_pop) and return an array of shape
        # (n_pop, dna_seq_len)
        if initializer:
            self.initializer = initializer
        else:
            self.initializer = lambda n_pop: np.random.rand(n_pop,self.dna_seq_len)
            
    def evolve(self,
               cost_function,
               max_gens = 200,
               n_pop = 50,
               mutation_rate = 0.2,
               min_cost = None,
               elitism = True):
        '''Launches genetic algorithm. Optional break criteria are:
            - max_gens: the maximum number of generations to complete before stopping.
            - n_pop: the population size of a generation
            - mutation_rate: the mutation rate, i.e. probability of introducing random changes into genes
            - min_cost: if cost of generation's top n genes falls below this value, stop evolution.
            - elitism: preserves the top 2 genes in each generation.'''
        
        # start evolution process
        for gen_i in range(max_gens):
            # get previous generation of gene configurations
            previous_gen_w_results = self._get_previous_gen(n_pop)
                        
            # produce this generation as offsprings of previous generation + mutation
            current_gen_wo_results = self._get_current_gen(previous_gen_w_results,
                                                           elitism,
                                                           mutation_rate)
                        
            # evaluate current gen and store in population history
            current_gen_w_results = self._eval_gen(cost_function,
                                                   current_gen_wo_results)
            
            # check min_cost break criterium if appropriate
            if min_cost != None:
                # if current generation is evolved enough, stop evolution
                #if self._is_current_gen_smart(current_gen_w_results,
                #                                 min_cost,
                #                                 crit='best'):
                #    break
                pass
                
        return current_gen_w_results
            
    def _get_previous_gen(self,
                          n_pop):
        '''Util function that obtains a generation from which the first generation of the evolution process
        will be generated. Either uses initialization or reads from population history.
        
        Returns a data frame shape (n_pop,dna_seq_length + 1) ,where the last column lists the scores of the genes.'''
        
        # population history shortcut
        pop_hist = self.population_history
        
        # column list shortcut
        dna_cols = ['gene'+str(i+1) for i in range(self.dna_seq_len)]
        
        # if this is the first evolution run, create initial population via intializer
        if str(type(pop_hist)) != "<class 'pandas.core.frame.DataFrame'>":
            # initialize population history
            self.population_history = pd.DataFrame(columns = ['n_gen'] + dna_cols + ['score'])
            # previous gen index
            n_previous_gen = -1
            # previous gen dna
            previous_gen_dna = self.initializer(n_pop)
            # arrange in frame
            previous_gen = pd.DataFrame(previous_gen_dna,
                                        columns = dna_cols)
            # attach bogus score to ensure first gen is random even though its the product of natural selection
            previous_gen['score'] = 1
            # set first generations index to 0, bc the current generation will up it by 1
            previous_gen['n_gen'] = 0
        # if not, obtain last population from history data frame
        else:
            n_previous_gen = int(max(pop_hist['n_gen']))
            previous_gen = pop_hist.loc[pop_hist['n_gen'] == n_previous_gen][['n_gen'] + dna_cols + ['score']]
            
        return previous_gen
    
    def _get_current_gen(self,
                         previous_gen_w_results,
                         elitism,
                         mutation_rate):
        ''' Util function that obtains the current geneation from the previous generation by
            - producing offsprings using crossover based on scores of previous generation
            - mutation by introducing random deviations to offspring genes.'''
            
        # get probability distribution from scores        
        p_genes = np.exp(previous_gen_w_results['score'].values) / np.sum(np.exp(previous_gen_w_results['score'].values))
        
        # get population size
        n_pop = previous_gen_w_results.shape[0]
        
        # initialize storage list for offsprings
        offspring_list = []
        
        #   column list shortcut
        dna_cols = ['gene'+str(i+1) for i in range(self.dna_seq_len)]
        
        # preserve 2 best genes if elitism is selected
        if elitism:
            sorted_previous_gen = previous_gen_w_results.sort_values('score',ascending=False)
            offspring_list.append(sorted_previous_gen[dna_cols].values[0])
            offspring_list.append(sorted_previous_gen[dna_cols].values[1])
                
        # iterate over parent pairs to produce 2 offsprings per 2 parents
        for i_couple in range(int(np.ceil(n_pop / 2))):
            # randomly sample to indices from previous population according to
            # distribution induced by its score
            i_father, i_mother = np.random.choice(range(n_pop),
                                                  2,
                                                  replace=False,
                                                  p = p_genes)
            # get father and mother
            father, mother = (previous_gen_w_results[dna_cols].values[i_father],
                              previous_gen_w_results[dna_cols].values[i_mother])
            
            # get two offsprings
            son, daughter = self._crossover(father,mother)
            
            # mutate offsprings
            x_son, x_daughter = self._mutate([son, daughter], mutation_rate)
            
            # add to offspring generation
            offspring_list += [x_son, x_daughter]
        
        # --- create offspring generation data frame
        #    cut off potential excess offspring if n_pop // 2 != 0
        offspring_list = offspring_list[:n_pop]
        # create empty data frame
        offspring_gen = pd.DataFrame(offspring_list,
                                     columns = dna_cols)            
        # set current generation index
        offspring_gen['n_gen'] = previous_gen_w_results['n_gen'][0] + 1
        # set empty score column
        offspring_gen['score'] = 0
        
        return offspring_gen
    
    def _crossover(self,
                   father,
                   mother):
        '''Util function that applies crossover to two parent genes to produce
        two offpsring genes. This is effectively calculating both versions of 
        the same convex combination of the parent genes, taken after some randomly
        smapled cut-off index.
        
        Takes two one dimensional arrays and returns two one dimensional arrays
        of the same length.'''
        
        # verify dimensions
        assert(len(father) == len(mother) == self.dna_seq_len)
        
        # --- get crossover specifications
        #   coefficients
        beta = np.random.uniform()
        cut_off = np.random.choice(self.dna_seq_len)
        
        # part of offpsrings that are inherited from single parent
        son_like_dad, daughter_like_mum = father[cut_off:], mother[cut_off:]
        
        # part of offsprings that are inherited from both parents
        son_like_parents = beta * father[:cut_off] + (1 - beta) * mother[:cut_off]
        daughter_like_parents = beta * mother[:cut_off] + (1 - beta) * father[:cut_off]
        
        # assemble offspring from parts
        son = np.concatenate([son_like_dad,son_like_parents])
        daughter = np.concatenate([daughter_like_mum,daughter_like_parents])
        
        # verify dimensions
        assert(len(son) == len(daughter) == self.dna_seq_len)
        
        return son, daughter
    
    def _mutate(self,
                genes,
                mutation_rate):
        '''Util function that applies mutation. Effectively just introduces random
        values into gene according to specified binomial probability.
        
        Takes a list of numpy arrays and returns a list of numpy arrays of the
        same length (list) and dimensions (arrays in list).'''
        
        x_genes = []
        
        # iterate over all genes to be mutated
        for gene in genes:
            # iterate over dna strains within one gene, i.e. scaler values
            for i_strain in range(self.dna_seq_len):
                # to mutate or not to mutate
                lets_mutate = np.random.choice([True, False],p=[mutation_rate,1 - mutation_rate])
                if lets_mutate:
                    # get rough specs of dna distribution to ensure mutation is not "malign"
                    gene_mean, gene_std = np.mean(gene), np.std(gene)
                    # mutate the current dna strain of current gene
                    gene[i_strain] = np.random.normal(loc=gene_mean,scale=gene_std)
                    
            x_genes.append(gene)
            
        return x_genes
    
    def _eval_gen(self,
                  cost_function,
                  current_gen):
        '''Util function that applies the cost function to each element of the 
        current population and stores the results in the population's data frame.'''
        
        # select gene data, i.e. the inputs to the cost function
        dna_cols = ['gene' + str(i+1) for i in range(self.dna_seq_len)]
        n_gen = current_gen['n_gen'][0]
        x = current_gen[dna_cols].values
        
        # assess genes via cost function
        scores = []
        n_pop = x.shape[0]
        for i,x_i in enumerate(x):
            print("Evaluating gene " + str(i) + " / " + str(n_pop) + " | Generation " + str(n_gen))
            score_i = cost_function(x_i.reshape((-1,self.dna_seq_len)))
            scores.append(score_i)
        
        # --- scores
        #   attach gene scores
        current_gen['score'] = scores
        #   rank by scores
        current_gen = current_gen.sort_values(by = 'score')
        #   attach to history
        self.population_history = pd.concat([self.population_history,current_gen])
        #   top of the crop
        print("Maximal score of " + str(max(scores)) + " achieved by gene " + str(np.argmax(scores)) + " | Generation " + str(n_gen))
        
        return current_gen
        
#----------------------------------------------------
# [11] define gene <-> network weight translator
#----------------------------------------------------

class GeneWeightTranslator(object):
    '''Object class handling the gene <-> weight conversion
    based on a neural network's individual parametrized layer. It stores the layer's weight's shapes
    and can be used both for initialization of the first generation of genes as well as 
    for later gene <-> network_weight conversion in the cost function.'''
    
    def __init__(self,
                 ffnetwork):
        # make sure network has been fixed already
        assert(ffnetwork.finalState)
        
        # --- attach network and network specific properties/objects
        #   network
        self.ffnetwork = ffnetwork
        #   layers with optimizable parameters
        self.weighted_layers = [weighted_layer for weighted_layer in ffnetwork.layers if weighted_layer.has_optimizable_params]
        #   shapes of optimizable parameters
        self.layer_weight_shapes = [(weighted_layer.Weight.shape,
                                     weighted_layer.bias.shape) for weighted_layer in (self.weighted_layers)]
        #   parameter counts of optimizable weights & biases
        self.layer_weight_sizes = [(np.prod(shape_w),
                                    np.prod(shape_b)) for (shape_w,shape_b) in self.layer_weight_shapes]
    
        print(self.layer_weight_sizes)
    
        #   dna sequence length = number of all optimizable parameters in network
        self.dna_seq_len = np.sum(self.layer_weight_sizes)
        
        # --- build gene initializer off of weighted layer's weight initializer methods
        #self.gene_initializer = self._get_gene_initializer()
        
    def get_current_weights(self):
        '''Util function that retrieves the current values of all optimizable parameters of the associated ffnetwork.
        Returns a list of tuples (Weight,bias).'''
        
        return [(weighted_layer.Weight,weighted_layer.bias) for weighted_layer in self.weighted_layers]
    
    def set_current_weights(self, weights):
        '''Util function that sets the associated ffnetwork's weights to the values specified in the 'weights'
        argument. Must be a list of (weight,bias) tuples of the correct shape.'''
        
        # --- check dimensions
        #   number of weighted layers
        assert(len(weights) == len(self.weighted_layers))
        #   dimensions of parameters specified
        for i,(weight, bias) in enumerate(weights):
            assert(weight.shape == self.layer_weight_shapes[i][0]) # weight shapes
            assert(bias.shape == self.layer_weight_shapes[i][1]) # bias shapes
            
        # -- set the network weights to specified values
        for weighted_layer, weight in zip(self.weighted_layers,weights):
            weighted_layer.Weight = weight[0] # weights are first tuple element
            weighted_layer.bias = weight[1] # bias are second tuple elements
        
    def gene_to_weights(self,
                        gene):
        '''Util function that converts a gene to a list of (Weight,bias) tuples,
        aligned with the GeneWeightTranslator's 'weighted_layers' attribute.'''
        
        # --- iterate over gene = [flat_weight_1:flat_bias_1:...:flat_weight_n,flat_bias_n],
        # --- cut off chunks and reshape
        converted_weights = []
        #   cut off chunks and reshape into weights and biases
        for (size_w, size_b),(shape_w,shape_b) in zip(self.layer_weight_sizes,self.layer_weight_shapes):
            # get chunk needed for current layer's parameters
            current_layer_chunk = gene[0,:size_w + size_b]
            # recreate weight
            weight = current_layer_chunk[:size_w].reshape(shape_w)
            bias = current_layer_chunk[size_w:].reshape(shape_b)
            
            # add reshaped layer parameters to storage list
            converted_weights.append((weight,bias))
            
            # update gene, i.e. drop used chunk
            gene = gene[0,size_w + size_b:].reshape((1,-1))
            
        return converted_weights
        
    def weights_to_gene(self,
                         weights):
        '''Util function that converts a list of (Weight,bias) tuples (aligned with
        the GeneWeightTranslator's 'weighted_layers' attribute) to a one dimensional
        numpy array, i.e. gene.'''
        
        # --- build one long gene from all (weight,bias) tuples
        #   initialize empty gene
        gene_segments = []
        for W,b in weights:
            W_flat, b_flat = W.reshape(1,-1), b.reshape(1,-1)
            gene_segments.append(W_flat)
            gene_segments.append(b_flat)
            
        gene = np.concatenate(gene_segments,axis=1)
        
        return gene[0]

    
    def initialize_genes(self,
                         n_pop):
        '''Util function that returns a batch of randomly initialized genes
        in the form of a numpy array of shape (n_pop,self.dna_seq_len).'''
        
        # --- create all n_pop genes indidvidually via assocdiated networks initiliazers
        init_genes_list = []
        
        for i_pop in range(n_pop):
            # randomly initialize all ffnetwork's weighted layers weights
            [weighted_layer.initializeWeightBias() for weighted_layer in self.weighted_layers]
            # grab initialized weights from associated ffnetwork
            init_weights = self.get_current_weights()
            # convert weights to gene
            init_gene = self.weights_to_gene(init_weights)
            # append gene to storage list
            init_genes_list.append(init_gene)
            
        init_genes = np.array(init_genes_list).reshape((n_pop,self.dna_seq_len))
        
        # final shape check
        assert(init_genes.shape == (n_pop, self.dna_seq_len))
        
        return init_genes
    
#----------------------------------------------------
# [12] define policy gradient for neural networks wrapper
#----------------------------------------------------
        
class PG(object):
    
    def __init__(self,
                 ffnetwork):
        
        self.ffnetwork = ffnetwork
        
        
    def train_network(self,
                      episode_generator,
                      n_episodes = 1000,
                      learning_rate = 0.01,
                      episode_batch_size = 10,
                      verbose = False,
                      display_steps = 1,
                      reward = 1,
                      regret = 1):
        '''Trains network using policy gradients based on samples produced by the 
        episode generator function. All eventual simulation are bundled into this
        magic function, which, for every training episode  (= sequence of (state,action) pairs
        up to first non-trivial reward) produces a batch of X, y and one reinforcement coefficient
        (usually +1 (encouraging behaviour displayed in corresponding sequence)/ -1 
        (discouraging behaviour displayed in corresponding sequence)).'''
        
        for i_episode in range(n_episodes):
            # psa
            if verbose:
                print("Running simulation to generate data | Episode " + str(i_episode+1) + " / " + str(n_episodes) + ".")
            # create this episodes training data and reinforcement coefficient
            X_ep, y_ep, r_ep = episode_generator(self.ffnetwork)
            # convert y_e to one hot encoded class labels
            Y_ep = self.ffnetwork.oneHotY(y_ep)
            # apply reward/regret scaling to reinforcment coefficient
            if r_ep < 0:
                r_ep *= regret
            elif r_ep > 0:
                r_ep *= reward
            else:
                print("Simulation returned trivial reward. No parameter update needed | Episode " + str(i_episode+1) + " / " + str(n_episodes) + ".")
                continue
            # --- train network on current batch
            n_batches = (X_ep.shape[0] // episode_batch_size) + 1
            if verbose:
                print("Processing simulation data and updating AI | Episode " + str(i_episode+1) + " / " + str(n_episodes) + ".")
            for i_batch in range(n_batches):
                # progress statement if required
                if ((i_batch+1) % display_steps) == 0 and verbose:
                    print('Episode: ',str(i_episode+1),'/',str(n_episodes))
                    print('Batch: ',str(i_batch+1),'/',str(n_batches))
                    print('---------------------------------------------------')
                #  forward prop
                _ = self.ffnetwork.forwardProp(X_ep)
                # backward prop including parameter updates
                self.ffnetwork.backwardProp(Y_ep,reinforcement_coeff=r_ep)
                
        return self.ffnetwork
    
    def save_network(self,save_dir,model_name,verbose = True):
        '''Save attached neural net model object as pickled file by calling the algorithm generic save function defined in section [1].'''
        
        return save_model(model = self.ffnetwork, save_dir = save_dir, model_name = model_name, verbose = verbose)
    
#----------------------------------------------------
# [13] define GLM class for one-dimensional regression/classification
#----------------------------------------------------

class GLM(object):
    
    def __init__(self,
                 family):
        
        # --- intialize some basic attributes
        # type of distribution assumed to be generating the response values
        self.family = family
        
        # canonical link & inverse link functions associated with specified family
        #self.link = self._get_canonical_link(family)
        self.inverse_link = self._get_inverse_canonical_link(family)
        
        family_functions = self._get_family_functions(family)
        self.b = family_functions['b']
        self.b_prime = family_functions['b_prime']
        self.c = family_functions['c']
        
        # weights (=beta) and bias (=beta_0)
        self.Weight = None
        self.bias = None
        
        # optimizer
        self.optimizer = self.most_recent_optimizer_used = None
        self.trained = False
        self.lossHistory = None
        self.classes_ordered = None
        
    def _get_canonical_link(self,
                            family):
        '''Helper function that returns the canonical link function associated with the given distribution type'''
        
        #assert family in ("poisson","normal","bernoulli","multi-bernoulli","gamma")
       # 
       # if (family in "poisson","gamma"):
        #    link = np.log
        #elif family == "normal":
        #    link = np.identity
        #elif family == "bernoulli":
        #    link = lambda eta: np.log(mu) - np.log(1 - mu) # logit
        #elif family == "multi-bernoulli":
            
            
        #return link
    
    def _get_inverse_canonical_link(self,
                                    family):
        '''Helper function that returns the inverse of the canonical link function associated with the given distribution type'''
        
        assert family in ("poisson","gaussian","bernoulli","multi-bernoulli","gamma")
        
        if (family in "poisson","gamma"):
            inverse_link = np.exp
        elif family == "gaussian":
            inverse_link = np.identity
        elif family == "bernoulli":
            inverse_link = lambda eta: np.exp(eta) / (1 + np.exp(eta)) # inverse logit = sigmoid
        elif family == "multi-bernoulli":
            inverse_link = softmax # vector-valued gradient of b
            
        return inverse_link
    
    def _get_family_functions(self,
                        family):
        '''Helper function that returns the b and c term of the specified member of the exponential distribution family.
        The b_prime is the same as the inverse link if the canonical link was chosen which is the only possible
        setting at this point, but I chose to define them separately in case I want to add support for custom
        link functions later.'''
        
        assert family in ("poisson")#,"normal","binomial","gamma")
        
        if family == "poisson":
            b = np.exp
            b_prime = np.exp
            c = lambda y, xi: 0
        elif family == 'gaussian':
            b = lambda theta: 1/2 * theta ** 2
            b_prime = np.identity
            c = lambda y, xi: 0
        elif family == 'bernoulli':
            b = lambda theta: np.log(1 + np.exp(theta))
            b_prime = lambda theta: np.exp(theta) / (1 + np.exp(theta))
            c = lambda y, xi: 0
        elif family == 'multi-bernoulli':
            b = lambda eta: np.log(np.sum(np.exp(eta),axis=1)) # heavily implied by the H2O docs; consistent with loglikelihood and choice of g^-1 = b'
            b_prime = softmax # vector valued gradient of b
            c = lambda y, xi: 0
            
        return {'b':b,'b_prime':b_prime,'c':c}

    def _get_one_hot_y(self,
			y):
        
        return getOneHotY(self,y)
        
    def forwardProp(self,
                     X,
                     apply_inverse_link=False):
        '''Helper function that applies the model parameters to the specified inputs and applies the link
        function (if desired; during training, the linear predictor is picked up by the backprop method as is)'''
        
        # ensure that weights have been initialized for this model instance
        assert self.Weight != None and self.bias != None
        
        # calculate linear predictor eta
        Eta = np.dot(X,self.Weight.T) + self.bias
                    
        if apply_inverse_link:
            P = self.inverse_link(Eta)
        elif (not apply_inverse_link):
            P = Eta
            
        return P
    
    def backwardProp(self,
                     X,
                     Eta,
                     Y):
        '''Helper function that calculates the gradient of the model's optimizable parameters self.Weights and self.bias
        and returns them in the common format Dcache.'''
        
        # calculate gradients for weights and bias
        # for multi-bernoulli, use vectorized version of parameter gradient calculation
        if self.family == "multi-bernoulli":
            batch_size = X.shape[0]
            beta_m = self.Weight.shape[0]
            DWeight = - 1/batch_size * np.sum(np.dot((Y - self.b_prime(Eta)).T,X),axis=0).reshape(beta_m,-1)
        else:
            DWeight = -np.mean(np.multiply(Y - self.b_prime(Eta),X),axis=0).reshape(1,-1)

        # bias gradient calculation is flexible
        Dbias = -np.mean(Y - self.b_prime(Eta),axis=0).reshape(1,-1)
        
        DCache = {'DWeight':DWeight,
                  'Dbias':Dbias}
        
        return DCache
    
    def initialize_optimizers(self,optimizer,eta,gamma,epsilon,lamda,batchSize):
        '''Creates and attaches optimizer objects to all network layers carrying optimizable parameters'''
        
        # make sure library supports specified optimization type
        assert optimizer in ['sgd']
        
        # get optimizer creator, i.e. 'prep optimizer cloning machine'
        if optimizer == 'sgd':
            optimizer_class = SGD
            
        # attach one optimizer to network for later reference
        self.optimizer = self.most_recent_optimizer_used = optimizer_class(eta,gamma,epsilon,lamda,batchSize)
        
    def initializeWeightBias(self,
                             n_predictors):
        '''Helper function that initializes model's weights and bias term.'''
        
        if str(type(model.classes_ordered)) == "<class 'NoneType'>":
            n_out = 1
        else:
            n_out = length(self.classes_ordered)

        self.Weight = np.random.randn(n_out,n_predictors) * 1 / (n_predictors + n_out)
        self.bias = np.ones((1,n_out))
        
    def loss(self,
             Eta,
             YBatch):
        '''Calculcates the loss function, i.e. the mean of the batch's negative log-likelihood.'''
        
        # NOTE: this needs to be updated to non-poisson dists - need to add the c term; alternatively,
        # add null&saturated model and display PoDE?
        
        # if using vectorized multi-bernoulli, collapse one-hot encoded arrays along column dimension
        if self.family == "multi-bernoulli":
            logloss_array = self.b(Eta) - sum(np.multiply(YBatch,Eta),axis=1)
        else:
            logloss_array = self.b(Eta) - np.multiply(YBatch,Eta)
            
        average_logloss = np.mean(logloss_array,axis=0)
        
        return average_logloss
        
    def updateGLMParams(self,Dcache,direction_coeff = 1):
        
        # retrieve updates for GLM's weights and bias using layer's optimizer
        DeltaWeight, DeltaBias = self.optimizer.get_parameter_updates(self.Weight,
                                                                      self.bias,
                                                                      Dcache)
            
        # update parameters with respective updates obtained from optimizer
        self.Weight += direction_coeff * DeltaWeight
        self.bias += direction_coeff * DeltaBias
        
    
    def trainGLM(self,
                     X,Y,
                     nEpochs=5,
                     batchSize=25,
                     optimizer='sgd',
                     eta=0.001,
                     gamma=0.99,
                     epsilon=0.0000001,
                     lamda=0,
                     displaySteps=50):
        '''Trains the generalized linear model using a specified type of gradient descent.'''
        
        # initialize storage for batch losses to be collected during training
        lossHistory = []
        recentLoss = 0
        
        # create and attach specified optimizers to layers, also keep one for later reference
        self.initialize_optimizers(optimizer,eta,gamma,epsilon,lamda,batchSize)
        
        # initialize one hot mapping if needed
        if self.family == "multi-bernoulli":
            self._get_one_hot_y(Y)
            n_classes = length(classes_ordered)
        else:
            n_classes = 1

        # initialize weights and bias term
        self.initializeWeightBias(n_predictors = X.shape[1])
        
        # execute training
        for epoch in range(nEpochs):
            for i,(XBatch,YBatch,nBatches) in enumerate(getBatches(X,Y,batchSize)):
                
                # apply one hot mapping if needed
                if self.family == "multi-bernoulli":
                    YBatch = self._get_one_hot_y(YBatch)

                # calculate linear predictor for loss function progress report
                Eta = self.forwardProp(XBatch,
                                       apply_inverse_link=False)
                
                # calculate this batch's loss function and update
                batchLoss = self.loss(Eta,YBatch)
                recentLoss += batchLoss
                
                # calculate gradients for model's optimizable parameters
                DCache = self.backwardProp(XBatch,
                                           Eta,
                                           YBatch)
                
                # update gradients using the specified method
                self.updateGLMParams(DCache)
                
                if ((i+1) % displaySteps) == 0:
                    averageRecentLoss = recentLoss / displaySteps
                    lossHistory.append(averageRecentLoss)
                    recentLoss = 0
                    print('Epoch: ',str(epoch+1),'/',str(nEpochs))
                    print('Batch: ',str(i+1),'/',str(nBatches))
                    print('Weight: ',str(self.Weight))
                    print('Loss averaged over last ',str(displaySteps),
                          ' batches: ',str(averageRecentLoss))
                    print('---------------------------------------------------')
        
        # announce end of training
        self.trained = True
        print('---------------------------------------------------')
        print('Training finished.')
        print('nEpochs:',str(nEpochs))
        print('Optimizer used:',str(self.most_recent_optimizer_used))
        print('batchSize:',str(batchSize))
        
        self.lossHistory = lossHistory
        
        return 0
    
    def save(self,save_dir,model_name,verbose = True):
        '''Save model object as pickled file by calling the algorithm generic save function defined in section [1].'''
        
        return save_model(model = self, save_dir = save_dir, model_name = model_name, verbose = verbose)

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:40:52 2019

@author: bettmensch
"""

import pickle
import numpy as np
import os,sys,inspect

# custom imports - sys path shenanigans needed
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # working dir
parentdir = os.path.dirname(currentdir) # lib dir of current repo: contains the game classes
deep_learning_dir = os.path.join(os.path.dirname(parentdir),"deep_learning_library") # dir in dl repo: contains dl classes

# diy deep learning lib from other repo
sys.path.append(deep_learning_dir)
from diy_deep_learning_library import FFNetwork

def create_and_prep_net(input_width,
                        input_height,
                        input_depth,
                        target_label_list):
    '''Util function that builds and preps a convolutional pilot net for training.'''
    # --- build conv net
    # build network
    neural_net = FFNetwork(2)
    
    kernelSize1 = 10
    channels1 = 2
    stride1 = 5
    padding1 = 'valid'
    
    poolingSize2 = 5
    stride2 = 2
    padding2 = 'valid'
    
    kernelSize3 = 5
    channels3 = 4
    stride3 = 1
    padding3 = 'valid'
    
    poolingSize4 = 2
    stride4 = 2
    padding4 = 'valid'
    
    n4 = 30
    
    n_output = 3
    
    neural_net.addConvLayer(kernelHeight=kernelSize1,
                       kernelWidth=kernelSize1,
                       channels=channels1,
                       stride=stride1,
                       padding=padding1,
                       activation='tanh')
    neural_net.addPoolingLayer(poolingHeight=poolingSize2,
                           poolingWidth=poolingSize2,
                           stride=stride2,
                           padding=padding2,
                           poolingType='max')
    neural_net.addConvLayer(kernelHeight=kernelSize3,
                           kernelWidth=kernelSize3,
                           channels=channels3,
                           stride=stride3,
                           padding=padding3,
                           activation='tanh')
    neural_net.addPoolingLayer(poolingHeight=poolingSize4,
                           poolingWidth=poolingSize4,
                           stride=stride4,
                           padding=padding4,
                           poolingType='max')
    
    neural_net.addFlattenConvLayer()
    
    neural_net.addFCLayer(n4,activation='tanh')
    
    neural_net.addFCLayer(n_output,activation='softmax')
    
    neural_net.fixateNetwork(np.zeros((10,input_depth,input_width,input_height)))
    
    print(neural_net)
    
    # prep network for inference without training it
    neural_net.oneHotY(np.array(target_label_list))

    neural_net.initialize_layer_optimizers('sgd',eta = 0.001,gamma = 0.99,epsilon = 0.00000001,lamda = 0,batchSize = 1)
    
    neural_net.trained = True
    
    return neural_net

def load_oldest_model(game,model_dir):
    '''Helper function that loads the most trained walker/pong model from specified model
    directory. Also retains the number of episodes the loaded model has been trained on.'''
    
    # sanity check inputs
    assert game in ('pong','walker')
    assert os.path.isdir(model_dir)
    
    # get oldest model for given game/experiment
    game_models = [model_name for model_name in os.listdir(model_dir) if game in model_name]
    
    if len(game_models) != 0:
        # get path to most trained model
        episodes_trained = list(map(lambda model_name: model_name.split('_')[2],game_models))
        pos = np.argmax(episodes_trained)
        game_model_name, episodes_trained = game_models[pos], episodes_trained[pos]
        game_model_path = os.path.join(model_dir,game_model_name)
        
        # load most trained model
        with open(game_model_path,'rb') as game_model_file:
            game_model = pickle.load(game_model_file)
    else:
        game_model, episodes_trained = None, 0
        
    return game_model, episodes_trained
        
        
    
    
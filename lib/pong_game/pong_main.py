# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:34:48 2019

@author: bettmensch
"""
# --- imports
# global imports
from pong_game_classes import Pong_with_AI, AI_Pong
from pong_settings import *
import numpy as np
import os,sys,inspect

# custom imports - sys path shenanigans needed
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # working dir
parentdir = os.path.dirname(currentdir) # lib dir of current repo: contains the game classes
deep_learning_dir = os.path.join(os.path.dirname(os.path.dirname(parentdir)),"deep_learning_library") # dir in dl repo: contains dl classes

sys.path.append(deep_learning_dir)

from diy_deep_learning_library import FFNetwork, PG


# --- main
def main():
    
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
    
    neural_net.fixateNetwork(np.zeros((10,N_CHANNELS,WINDOW_SIZE[0],WINDOW_SIZE[1])))
    
    print(neural_net)
    
    # prep network for inference without training it
    neural_net.oneHotY(np.array([UP,DOWN,NONE]))

    neural_net.initialize_layer_optimizers('sgd',eta = 0.001,gamma = 0.99,epsilon = 0.00000001,lamda = 0,batchSize = 1)
    
    neural_net.trained = True
    
    # --- create policy gradient wrapper
    # create epsiode generator function
    def ai_pong_episode_generator(ai_network):
        
        ai_pilot = AI_Pong(ai_network)
        
        #ai_log = Pong_with_AI().start(ai_pilot = None).log
        ai_log = Pong_with_AI().start(ai_pilot = ai_pilot).log
        
        X = np.concatenate(ai_log['X'],axis=0)
        y = np.array(ai_log['y']).reshape(-1,1)
        ri_coeff = ai_log['reinforce_coeff']
        
        return X,y,ri_coeff
    
    # if reloading trained pilot, set this to number of episodes already trained
    pages_trained = 13000
    
    for training_chapter in range(TRAINING_CHAPTERS):
        
        # create pg object with above episode generator and neural net
        policy_gradient_pong = PG(neural_net)
    
        # --- train network with policy gradient
        policy_gradient_pong.train_network(episode_generator = ai_pong_episode_generator,
                                           n_episodes = TRAINING_PAGES,
                                           learning_rate = 0.01,
                                           episode_batch_size = 10,
                                           verbose = True,
                                           reward = 1,
                                           regret = 1)

        # save checkpoint model ~ AI
        neural_net = policy_gradient_pong.ffnetwork
        neural_net_name = 'pong_pilot_' + str((training_chapter+1) * TRAINING_PAGES + pages_trained) + '_episodes'
        neural_net.save(save_dir = PONG_MODEL_DIR, model_name = neural_net_name)
    
if __name__ == "__main__":
    pg_pong = main()
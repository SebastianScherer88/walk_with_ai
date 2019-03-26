# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:58:02 2019

@author: bettmensch
"""

from game_class import Walk_With_AI, AI_Walker
from settings import *
import numpy as np
from diy_dl_copy import FFNetwork, PG

def main():
    
    # --- build conv net
    # build network
    neuralNet = FFNetwork(2)
    
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
    
    n4 = 336
    
    n_output = 3
    
    neuralNet.addConvLayer(kernelHeight=kernelSize1,
                       kernelWidth=kernelSize1,
                       channels=channels1,
                       stride=stride1,
                       padding=padding1,
                       activation='tanh')
    neuralNet.addPoolingLayer(poolingHeight=poolingSize2,
                           poolingWidth=poolingSize2,
                           stride=stride2,
                           padding=padding2,
                           poolingType='max')
    neuralNet.addConvLayer(kernelHeight=kernelSize3,
                           kernelWidth=kernelSize3,
                           channels=channels3,
                           stride=stride3,
                           padding=padding3,
                           activation='tanh')
    neuralNet.addPoolingLayer(poolingHeight=poolingSize4,
                           poolingWidth=poolingSize4,
                           stride=stride4,
                           padding=padding4,
                           poolingType='max')
    
    neuralNet.addConvToFCReshapeLayer(n4)
    
    neuralNet.addFCLayer(n4,activation='tanh')
    
    neuralNet.addFCLayer(n_output,activation='softmax')
    
    neuralNet.fixateNetwork(np.zeros((10,N_CHANNELS,WINDOW_SIZE[0],WINDOW_SIZE[1])))
    
    print(neuralNet)
    
    # prep network for inference without training it
    neuralNet.oneHotY(np.array([LEFT,NONE,RIGHT]))

    neuralNet.initialize_layer_optimizers('sgd',eta = 0.001,gamma = 0.99,epsilon = 0.00000001,lamda = 0,batchSize = 1)
    
    neuralNet.trained = True
    
    # --- create policy gradient wrapper
    # create epsiode generator function
    def ai_walker_episode_generator(ai_network):
        
        ai_pilot = AI_Walker(ai_network)
        
        ai_log = Walk_With_AI().start(ai_pilot = ai_pilot).log
        
        X = np.concatenate(ai_log['X'],axis=0)
        y = np.concatenate(ai_log['y'],axis=0).reshape(-1,1)
        ri_coeff = ai_log['reinforce_coeff']
        
        return X,y,ri_coeff
    
    # create pg object with above episode generator and neural net
    policy_gradient_walker = PG(neuralNet)
    
    # --- train network with policy gradient
    policy_gradient_walker.train_network(episode_generator = ai_walker_episode_generator,
                                         n_episodes = 100,
                                         learning_rate = 0.01,
                                         episode_batch_size = 10,
                                         verbose = False,
                                         reward = 1,
                                         regret = 1)
        
    return neural_ai_with_log.log
    
if __name__ == "__main__":
    log = main()
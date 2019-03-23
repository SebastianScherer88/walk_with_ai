# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:58:02 2019

@author: bettmensch
"""

from game_class import Walk_With_AI, AI_Walker
from settings import *
from diy_dl_copy import FFNetwork

def main():
    
    # --- build conv net
    # build network
    neuralNet = diy_deep_learning_library.FFNetwork(2)
    
    kernelSize1 = 5
    channels1 = 2
    stride1 = 1
    padding1 = 'valid'
    
    poolingSize2 = 2
    stride2 = 2
    padding2 = 'valid'
    
    kernelSize3 = 5
    channels3 = 4
    stride3 = 1
    padding3 = 'valid'
    
    poolingSize4 = 2
    stride4 = 2
    padding4 = 'valid'
    
    n4 = 40
    
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
    
    neuralNet.addConvToFCReshapeLayer(n5)
    
    neuralNet.addFCLayer(n4,activation='tanh')
    
    neuralNet.addFCLayer(n_output,activation='softmax')
    
    neuralNet.fixateNetwork(np.zeros((10,N_CHANNELS,WINDOW_SIZE[0],WINDOW_SIZE[1])))
    
    # --- create pilot wrapper around conv net
    
    
    # --- let conv ent pilot the game
    new_game = Walk_With_AI()
    
    conv_pilot = new_game.start()
        
    return conv_pilot.log
    
if __name__ == "__main__":
    log = main()
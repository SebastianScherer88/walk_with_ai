# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:40:52 2019

@author: bettmensch
"""

import pickle
import os

def load_oldest_model(game,model_dir):
    '''Helper function that loads the most trained walker/pong model from specified model
    directory. Also retains the number of episodes the loaded model has been trained on.'''
    
    # sanity check inputs
    assert game in ('pong','walker')
    assert os.path.isdir(model_dir)
    
    # get oldest model for given game/experiment
    os.listdir(model_dir)
    
    
    
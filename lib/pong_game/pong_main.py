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

# game generic util functions
sys.path.append(parentdir)
from util import create_and_prep_net, load_oldest_model

# diy deep learning lib from other repo
sys.path.append(deep_learning_dir)
from diy_deep_learning_library import FFNetwork, PG

# create episode generator function
def ai_pong_episode_generator(ai_network):
    
    ai_pilot = AI_Pong(ai_network)
    
    #ai_log = Pong_with_AI().start(ai_pilot = None).log
    ai_log = Pong_with_AI().start(ai_pilot = ai_pilot).log
    
    X = np.concatenate(ai_log['X'],axis=0)
    y = np.array(ai_log['y']).reshape(-1,1)
    ri_coeff = ai_log['reinforce_coeff']
    
    return X,y,ri_coeff

# create Pong training function
def teach_pong(seasons = 5,
               episodes_per_season = 100,
               from_scratch = False,
               model_dir = PONG_MODEL_DIR):
    '''Teaches the game Pong to convolutional net based AI for the specified
    number of episodes and seasons. Can pick up trained models to continue training,
    or train a new one from scratch. Saves model after each season.'''
    
    # --- get model
    if not from_scratch:
        # load oldest model if possible
        neural_net, taught_episodes = load_oldest_model(game = 'pong', model_dir = '')
    else:
        neural_net = None
    
    # if training from scratch is desired, or if above model loading was unsuccessful, get new model
    if neural_net == None:
        neural_net = create_and_prep_net(input_width = WINDOW_SIZE[0],
                                         input_height = WINDOW_SIZE[1],
                                         input_depth = N_CHANNELS,
                                         target_label_list = [UP,DOWN,NONE])
    
    # --- train model
    for season in range(N_TRAINING_SEASONS):
        
        # create pg object with above episode generator and neural net
        policy_gradient_pong = PG(neural_net)
    
        # train network with policy gradient
        policy_gradient_pong.train_network(episode_generator = ai_pong_episode_generator,
                                           n_episodes = N_TRAINING_EPISODES,
                                           learning_rate = 0.01,
                                           episode_batch_size = 10,
                                           verbose = True,
                                           reward = 1,
                                           regret = 1)

        # save checkpoint model ~ AI
        neural_net = policy_gradient_pong.ffnetwork
        neural_net_name = '_'.join(['pong_pilot', str((season + 1) * N_TRAINING_EPISODES), 'epsiodes'])
        neural_net.save(save_dir = PONG_MODEL_DIR, model_name = neural_net_name)
    
if __name__ == "__main__":
    main()
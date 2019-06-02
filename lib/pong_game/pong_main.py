# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:34:48 2019

@author: bettmensch
"""
# --- imports
# global imports
from pong_game_classes import Pong_with_AI, AI_Pong
from functools import partial
from pong_settings import *
import numpy as np
import os,sys,inspect

# custom imports - sys path shenanigans needed
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # working dir
parentdir = os.path.dirname(currentdir) # lib dir of current repo: contains the game classes
deep_learning_dir = os.path.join(os.path.dirname(os.path.dirname(parentdir)),"deep_learning_library") # dir in dl repo: contains dl classes

# game generic util functions
sys.path.append(parentdir)
from util import create_and_prep_net, load_models, get_teach_command_line_args

# diy deep learning lib from other repo
sys.path.append(deep_learning_dir)
from diy_deep_learning_library import FFNetwork, PG

# create episode generator function
def ai_pong_episode_generator(ai_network, visual, net_type):
    
    ai_pilot = AI_Pong(ai_network,net_type)
    
    #ai_log = Pong_with_AI().start(ai_pilot = None).log
    ai_log = Pong_with_AI().start(ai_pilot = ai_pilot,
                         visual = visual,
                         level_history_type = net_type).log
    
    X = np.concatenate(ai_log['X'],axis=0)
    y = np.array(ai_log['y']).reshape(-1,1)
    ri_coeff = ai_log['reinforce_coeff']
    
    return X,y,ri_coeff

# create Pong training function
def teach_pong(seasons = N_TRAINING_SEASONS,
               episodes_per_season = N_TRAINING_EPISODES,
               from_scratch = False,
               net_type = 'conv',
               model_dir = '',
               visual = True):
    '''Teaches the game Pong to convolutional net based AI for the specified
    number of episodes and seasons. Can pick up trained models to continue training,
    or train a new one from scratch. Saves model after each season.'''
    
    assert (net_type in ('conv','mlp'))
    
    # --- get model
    if not from_scratch:
        # load oldest model if possible
        neural_net, taught_episodes = load_models(game = 'pong',
                                                  net_type = net_type,
                                                  model_dir = model_dir)
    else:
        neural_net = None
    
    # if training from scratch is desired, or if above model loading was unsuccessful, get new model
    if neural_net == None:
        if net_type == 'conv':
            input_width = WINDOW_SIZE[0]
        elif net_type == 'mlp':
            input_width = POSITIONAL_FEATURES_N
            
        neural_net = create_and_prep_net(net_type = net_type,
                                         input_width = input_width,
                                         input_height = WINDOW_SIZE[1],
                                         input_depth = N_CHANNELS,
                                         target_label_list = [UP,DOWN,NONE])
        taught_episodes = 0
    
    # --- train model
    for season in range(seasons):
        
        # create pg object with above episode generator and neural net
        policy_gradient_pong = PG(neural_net)
    
        # train network with policy gradient
        policy_gradient_pong.train_network(episode_generator = partial(ai_pong_episode_generator,
                                                                       visual = visual,
                                                                       net_type = net_type),
                                           n_episodes = episodes_per_season,
                                           learning_rate = 0.01,
                                           episode_batch_size = 10,
                                           verbose = True,
                                           reward = 1,
                                           regret = 1)

        # save checkpoint model ~ AI
        neural_net = policy_gradient_pong.ffnetwork
        epsiodes_so_far = taught_episodes + (season + 1) * episodes_per_season
        neural_net_name = '_'.join(['pong_pilot',net_type, str(epsiodes_so_far), 'episodes'])
        neural_net.save(save_dir = model_dir, model_name = neural_net_name)
        
def main():
    
    parsed_args = get_teach_command_line_args(season_default = N_TRAINING_SEASONS,
                                              episode_default = N_TRAINING_EPISODES)
    
    # ensure specified model directory exists
    assert (os.path.exists(parsed_args['model_directory']))
    
    print(parsed_args)
    
    teach_pong(seasons = parsed_args['n_seasons'],
               episodes_per_season = parsed_args['n_episodes'],
               from_scratch = parsed_args['train_from_scratch'],
               net_type = parsed_args['net_type'],
               model_dir = parsed_args['model_directory'],
               visual = parsed_args['visual_mode'])
    
if __name__ == "__main__":
    main()
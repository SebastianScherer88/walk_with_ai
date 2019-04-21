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
from util import create_and_prep_net, load_oldest_model, get_command_line_args

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
def teach_pong(seasons = N_TRAINING_SEASONS,
               episodes_per_season = N_TRAINING_EPISODES,
               from_scratch = False,
               model_dir = PONG_MODEL_DIR):
    '''Teaches the game Pong to convolutional net based AI for the specified
    number of episodes and seasons. Can pick up trained models to continue training,
    or train a new one from scratch. Saves model after each season.'''
    
    # --- get model
    if not from_scratch:
        # load oldest model if possible
        neural_net, taught_episodes = load_oldest_model(game = 'pong', model_dir = model_dir)
    else:
        neural_net = None
    
    # if training from scratch is desired, or if above model loading was unsuccessful, get new model
    if neural_net == None:
        neural_net = create_and_prep_net(input_width = WINDOW_SIZE[0],
                                         input_height = WINDOW_SIZE[1],
                                         input_depth = N_CHANNELS,
                                         target_label_list = [UP,DOWN,NONE])
        taught_episodes = 0
    
    # --- train model
    for season in range(seasons):
        
        # create pg object with above episode generator and neural net
        policy_gradient_pong = PG(neural_net)
    
        # train network with policy gradient
        policy_gradient_pong.train_network(episode_generator = ai_pong_episode_generator,
                                           n_episodes = episodes_per_season,
                                           learning_rate = 0.01,
                                           episode_batch_size = 10,
                                           verbose = True,
                                           reward = 1,
                                           regret = 1)

        # save checkpoint model ~ AI
        neural_net = policy_gradient_pong.ffnetwork
        epsiodes_so_far = taught_episodes + (season + 1) * episodes_per_season
        neural_net_name = '_'.join(['pong_pilot', str(epsiodes_so_far), 'epsiodes'])
        neural_net.save(save_dir = model_dir, model_name = neural_net_name)
        
def main():
    
    parsed_args = get_command_line_args(season_default = N_TRAINING_SEASONS,
                                        episode_default = N_TRAINING_EPISODES)
    
    print(parsed_args)
    
    teach_pong(from_scratch = parsed_args['train_from_scratch'],
               seasons = parsed_args['n_seasons'],
               episodes_per_season = parsed_args['n_episodes'])
    
if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:44:38 2019

@author: bettmensch
"""

import pickle
import os

# set working directory
os.chdir('C:\\Users\\bettmensch\\GitReps\\walk_with_ai\\lib\\pong_game')

# ---import dependencies
# global imports
from pong_game_classes import Pong_with_AI, AI_Pong
from functools import partial
from pong_settings import *
import numpy as np
import sys,inspect

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

n = 50

with open('C:\\Users\\bettmensch\\GitReps\\walk_with_AI\\models\\pong_pilot_mlp_'+str(n)+'_episodes','rb') as model_file:
    model = pickle.load(model_file)
    
ai_pilot = AI_Pong(model = model,net_type = 'mlp',sample_from_distribution=False)

#ai_log = Pong_with_AI().start(ai_pilot = None).log
for i in range(5):
    ai_log = Pong_with_AI(frames_per_second = 20).start(ai_pilot = ai_pilot, visual = True, max_frames=500, level_history_type = 'mlp').log
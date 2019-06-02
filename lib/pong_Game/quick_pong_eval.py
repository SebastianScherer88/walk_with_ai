# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:44:38 2019

@author: bettmensch
"""

import pickle

n = 415

with open('C:\\Users\\bettmensch\\GitReps\\walk_with_AI\\models\\pong_pilot_mlp_'+str(n)+'_episodes','rb') as model_file:
    model = pickle.load(model_file)
    
ai_pilot = AI_Pong(model)

#ai_log = Pong_with_AI().start(ai_pilot = None).log
for i in range(5):
    ai_log = Pong_with_AI(frames_per_second = 20).start(ai_pilot = ai_pilot, visual = True, max_frames=500).log
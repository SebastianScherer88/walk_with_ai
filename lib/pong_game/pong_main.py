# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:34:48 2019

@author: bettmensch
"""

from pong_game_classes import Pong
from pong_settings import *

def main():
    
    pong_game = Pong()
    
    n_second = 100
    
    pong_game.start(max_frames = FRAMES_PER_SECOND * n_second)
    
    
if __name__ == "__main__":
    main()
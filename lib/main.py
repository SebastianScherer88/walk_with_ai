# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:58:02 2019

@author: bettmensch
"""

from game_class import Walk_With_AI

def main():
    
    new_game = Walk_With_AI()
    
    raw_level_history = new_game.start()
        
    return raw_level_history
    
if __name__ == "__main__":
    level_history = main()
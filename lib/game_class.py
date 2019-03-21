# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:21:09 2019

Game class where character has to navigate obstacle course from designated start
to designated finish. Can be controlled by player or by specified AI.

@author: bettmensch
"""

import pygame as pg
from pygame.sprite import Sprite
from pygame.math import Vector2D as vec
from settings import *

class Walker(Sprite):
    
    def __init__(self,
                 image_surface,
                 pos_x,
                 pos_y,
                 theta,
                 *groups):
        
        # initialize sprite class instance
        Sprite.__init__(self,
                        *groups)
        
        # set positional and orientational attributes
        self.x = vec(pos_x,pos_y)
        self.v = vec(0,SPEED).rotate(theta)
        
        # set image and rect attributes
        self.image = pg.transform.rotate(image_surface,theta)
        self.rect = image_surface.get_rect()
        


class Walk_With_AI(object):
    
    def __init__(self,
                 frames_per_second,
                 window_size = (400,400))
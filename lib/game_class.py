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
        self.v = vec(0,D_X).rotate(theta)
        self.theta = theta
        
        # set image and rect attributes
        self.original_image = image_surface
        
        # initialize position attributes by first call to update()
        self.update(turn = NONE)

    def update(self,
               turn):
        
        # sanity check commands
        assert turn in (LEFT,NONE,RIGHT)
        
        # update angle according to steer
        if turn == LEFT:
            d_theta = D_THETA
        elif turn == RIGHT:
            d_theta = -D_THETA
        elif turn == NONE:
            d_theta = 0
            
        # update angle and velocity according to steer
        self.theta += d_theta
        self.v = self.v.rotate(d_theta)
        
        # get new position with updated velocity
        self.x += self.v
        self.rect.center = (int(self.x.x, int(self.x.y)))
        
        # update image surface orientation as needed
        if turn != NONE:
            self.image = pg.transform.rotate(self.original_image,
                                             self.theta)
            self.rect = self.image.get_rect()
        
class Walk_With_AI(object):
    
    def __init__(self,
                 frames_per_second,
                 window_size = WINDOW_SIZE):
        
        # initialize pygame window
        pg.init()
        
        # get game clock
        self.clock = pg.time.Clock()
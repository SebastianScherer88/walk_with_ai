# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:47:34 2019

This script contains the Pong game class, the AI_Ponger class and the feature
extractor used for the pong with ai run.

@author: bettmensch
"""

import sys
import pygame as pg
import numpy as np
from pygame.sprite import Sprite
from pygame.math import Vector2 as vec
from pong_settings import *
from functools import partial

class Paddle(object):
    
    def __init__(self,
                 *groups,
                 pos_x = 0,
                 pos_y = 0,
                 width = PADDLE_WIDTH,
                 length = PADDLE_LENGTH,
                 speed = PADDLE_SPEED,
                 color = WHITE):
        
        # create base sprite instance
        Sprite.__init__(self,
                        *groups)
        
        # create visual attributes
        self.image = pg.Surface(width,length)
        self.rect = self.image.fill(WHITE)
        self.rect.topleft = (pos_x,pos_y)
        
        # create positional attributes
        self.x = vec(pos_x,pos_y)
        self.v_up = vec(0,-speed)
        
    def update(self,
               move):
        
        assert move in (UP,DOWN,NONE)
        
        if move == NONE:
            v = 0 * self.v_up
        elif move == UP:
            v  = self.v_up
        elif move == DOWN:
            v = -self.v_up
        
        self.x += v
        self.rect.topleft = (int(self.x.x),int(self.x.y))
        
class Ball(object):
    
    def __init__(self,
                 *groups,
                 pos_x = 0,
                 pos_y = 0,
                 v_x = np.sqrt(2),
                 v_y= np.sqrt(2),
                 radius = BALL_RADIUS,
                 color = WHITE,
                 background = BLACK,
                 window_size = WINDOW_SIZE):
        
        # create base sprite instance
        Sprite.__init__(self,
                        *groups)
        
        self.window_size = WINDOW_SIZE
        
        # create visual attributes
        self.image = pg.Surface(2*radius,2*radius)
        self.rect = self.image.fill(background)
        self.rect.topleft = (pos_x,pos_y)
        
        # add circle to image, make edges transparent
        pg.draw.circle(self.image,color,(radius,radius),radius)
        self.image.set_colorkey(color)
        
        # create positional attributes
        self.speed = BALL_SPEED
        self.x = vec(pos_x,pos_y)
        
        self.v = vec(v_x,v_y)
        self.v = BALL_SPEED * self.v / self.v.get_length()
        
    def update(self):
        '''Only checks for out of bounds w.r.t height. The collisions with
        paddles are handled in separate method.'''
        
        # if too high, bring back down and bounce
        if self.rect.top < 0:
            self.x.y = 0
            self.v.y = -self.v.y
            # or if too low, bring back up and bounce
        elif self.rect.bottom > self.window_size[1]:
            self.x.y = self.window_size[1]
            self.v.y = -self.v.y
        
        self.x += self.v
        self.rect.topleft = (int(self.x.x),int(self.x.y))
        
    def bounce_off_paddle(self,
                          paddle,
                          min_angle_factor = MIN_BOUNCE_ANGLE_FACTOR):
        '''Check for collisions with paddle and bounce of in an appropriate manner.
        The outgoing bounce angle is determined by the collision point's relative
        position to the bouncing paddle's center, while speed is kept constant.'''
        
        # get outbound angle
        d_norm = min_angle_factor * (self.y - paddle.rect.centerx) / (paddle.rect.height/2 + self.rect.height) # normalize relative position to [-1,1]
        alpha_out = np.arccos(d_norm) # get
        
        # adjust velocity ofr horizontal & vertical bounce
        sign_x = np.sign(self.v.x)
        
        self.v.y = BALL_SPEED * np.cos(alpha_out)
        self.v.x = sign_x * (-1) * BALL_SPEED * np.sin(alpha_out)
        
        # adjust position
        if self.rect.centerx > self.window_size[0] / 2:
            self.rect.right = paddle.rect.left
        elif self.rect.centerx < self.window_size[0] / 2:
            self.rect.left = paddle.rect.right
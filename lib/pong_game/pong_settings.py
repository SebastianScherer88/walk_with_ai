# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:48:51 2019

This script sets the defaults for pong with ai variables.

@author: bettmensch
"""

import pygame as pg
import numpy as np

# size of game window in pixel
WINDOW_SIZE = (600,450)
FRAMES_PER_SECOND = 30

# colors
WHITE = pg.Color(255,255,255) # used for ball and paddles
BLACK = pg.Color(0,0,0) # used for background

# pong paddle specs
PADDLE_WIDTH = 10
PADDLE_LENGTH = 70
PADDLE_SPEED = 8 # speed in vertical direction in pixel/frame
PADDLE_INSET_RATIO = 0.05
OPPONENT_STATIONARY_MARGIN = 0.25

# pong direction options
UP = 'UP'
DOWN = 'DOWN'
NONE = 'NONE'

# ball specs
BALL_RADIUS = 8
BALL_SPEED = 12 # ball speed in pixel/frame
MIN_BOUNCE_ANGLE_FACTOR = 2 # 0.8 ~ 10 degrees
BALL_INITIAL_MAX_ANGLE = np.pi/4

# level state options
WON = 'WON'
LOST = 'LOST'
TIMEDOUT = 'TIMEDOUT'
CONTINUE = 'CONTINUE'
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:48:51 2019

This script sets the defaults for pong with ai variables.

@author: bettmensch
"""

import pygame as pg
import numpy as np

# size of game window in pixel
WINDOW_SIZE = (200,180)
FRAMES_PER_SECOND = 200

# colors
WHITE = pg.Color(255,255,255) # used for ball and paddles
BLACK = pg.Color(0,0,0) # used for background

# pong paddle specs
PADDLE_WIDTH = 5
PADDLE_LENGTH = 25
PADDLE_SPEED = 4.5 # speed in vertical direction in pixel/frame
PADDLE_INSET_RATIO = 0.05
OPPONENT_STATIONARY_MARGIN = 0.25

# pong direction options
UP = 'UP'
DOWN = 'DOWN'
NONE = 'NONE'

# ball specs
BALL_RADIUS = 4
BALL_SPEED = 6 # ball speed in pixel/frame
MIN_BOUNCE_ANGLE_FACTOR = 2 # 0.8 ~ 10 degrees
BALL_INITIAL_MAX_ANGLE = np.pi/4

# level state options
WON = 'WON'
LOST = 'LOST'
TIMEDOUT = 'TIMEDOUT'
CONTINUE = 'CONTINUE'

# ai specs
REWARDS_MAP = {WON: 1,
               LOST: 0,
               TIMEDOUT: 0}
N_CHANNELS = 3

N_TRAINING_SEASONS = 4
N_TRAINING_EPISODES = 3000

PONG_MODEL_DIR = 'C:\\Users\\bettmensch\\GitReps\\walk_with_AI\\models'

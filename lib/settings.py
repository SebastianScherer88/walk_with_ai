# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:20:44 2019

@author: bettmensch
"""

import pygame as pg

# size of game window in pixel
WINDOW_SIZE = (400,400)

# window color
WHITE = pg.Color(255,255,255)

# frames per second
FRAMES_PER_SECOND = 10

# target size
MARKER_SIZE = 20

# border thickness
BORDER = 10

# border color
BORDER_COLOR = (100,100,100)

# WALKER color
WALKER_IMAGE_PATH = "../image/walker_image.bmp"

# marker color
START_COLOR = (0,50,200)
FINISH_COLOR = (200,0,0)

# level meta data: positions and sizes of bounding course boxes
LEVELS = [{'blocks':[{'x':0,'y':0,'w':WINDOW_SIZE[0],'h':BORDER},
                     {'x':0,'y':WINDOW_SIZE[1]-BORDER,'w':WINDOW_SIZE[0],'h':BORDER},
                     {'x':int(WINDOW_SIZE[0]/2),'y':0,'w':BORDER,'h':int(WINDOW_SIZE[1]/2)}],
            'start':{'x':10,'y':int(WINDOW_SIZE[0]/4),'w':MARKER_SIZE,'h':MARKER_SIZE,'color':START_COLOR},
            'finish':{'x':WINDOW_SIZE[0]-MARKER_SIZE-BORDER,'y':int(WINDOW_SIZE[0]/4),'w':MARKER_SIZE,'h':MARKER_SIZE,'color':FINISH_COLOR}
            }]

# speed of walker in pixel/ frame
D_X = 2

# angular speed in degree / frame
D_THETA = 10

# quit event flag
QUIT = "QUIT"

# level state flag
WON = 'WON'
LOST = 'LOST'
CONTINUE = 'CONTINUE'

# directional instructions for walker
LEFT = 'LEFT'
NONE = 'NONE'
RIGHT = 'RIGHT'
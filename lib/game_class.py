# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:21:09 2019

Game class where character has to navigate obstacle course from designated start
to designated finish. Can be controlled by player or by specified AI.

@author: bettmensch
"""

import sys
import pygame as pg
import numpy as np
from pygame.sprite import Sprite
from pygame.math import Vector2 as vec
from settings import *

class Walker(Sprite):
    
    def __init__(self,
                 *groups,
                 image_surface,
                 pos_x = 0,
                 pos_y = 0,
                 theta = 0):
        
        # initialize sprite class instance
        Sprite.__init__(self,
                        *groups)
        
        # set positional and orientational attributes
        self.x = vec(pos_x,pos_y)
        self.v = vec(D_X,0).rotate(theta)
        self.theta = theta
        
        # set image and rect attributes
        self.original_image = image_surface
        self.original_image.set_colorkey(WHITE)
        
        # initialize position attributes by first call to update()
        self.update(turn = NONE)

    def update(self,
               turn):
        
        # sanity check commands
        #print("Turn:",turn)
        
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
        self.v = self.v.rotate(-d_theta)
        
        # get new position with updated velocity
        self.x += self.v
        
        # update image surface orientation as needed
        self.image = pg.transform.rotate(self.original_image,
                                         self.theta)
        self.rect = self.image.get_rect()
        self.rect.center = (int(self.x.x), int(self.x.y))
            
class Block(Sprite):
    
    def __init__(self,
                 *groups,
                 x,
                 y,
                 w,
                 h,
                 color = BORDER_COLOR):
        
        # initialize base class instance and add to groups
        Sprite.__init__(self,*groups)
        
        # set attributes
        self.image = pg.Surface((w,h))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)
        
        
class Walk_With_AI(object):
    
    def __init__(self,
                 frames_per_second = FRAMES_PER_SECOND,
                 window_size = WINDOW_SIZE,
                 levels = LEVELS,
                 walker_image_path = WALKER_IMAGE_PATH):
        
        # initialize pygame window
        pg.init()
        
        # get game clock
        self.clock = pg.time.Clock()
        self.fps = frames_per_second
        
        # some game params
        self.window_size = window_size
        self.levels = levels
        self.walker_image = pg.image.load(walker_image_path)
        
    def get_level_sprites(self,
                          level):
        
        # sanity check
        for essential_key in 'blocks','start','finish':
            assert essential_key in list(level.keys())
        
        # create group for all sprites
        all_sprites = pg.sprite.Group()
        
        # create boundary sprite group
        blocks = pg.sprite.Group()
        [Block(all_sprites,blocks,**block_specs) for block_specs in level['blocks']]
        
        # add marker sprites and create finish sprite group
        finish = pg.sprite.Group()
        Block(all_sprites,**level['start'])
        Block(all_sprites,finish,**level['finish'])
        
        # create player sprite group
        walker = pg.sprite.Group()
        walker = Walker(all_sprites,
                        image_surface = self.walker_image,
                        pos_x = level['start']['x'],
                        pos_y = level['start']['y'],
                        theta = 0)
        
        return all_sprites,blocks, finish, walker
    
    def get_player_steer(self):
                    
        keys_pressed = pg.key.get_pressed()
        
        if (keys_pressed[pg.K_LEFT] and keys_pressed[pg.K_RIGHT]) or (not keys_pressed[pg.K_LEFT] and not keys_pressed[pg.K_RIGHT]):
            steer = NONE
        elif (keys_pressed[pg.K_LEFT] and not keys_pressed[pg.K_RIGHT]):
            steer = LEFT
        elif (not keys_pressed[pg.K_LEFT] and keys_pressed[pg.K_RIGHT]):
            steer = RIGHT
                         
        return steer
    
    def get_ai_steer_and_log(self,ai_pilot,raw_level_history,level_state):
            
        # create input to ai pilot from raw level states
        ai_input = ai_pilot.create_input(raw_level_history)
        
        ai_steer = ai_pilot.create_output(ai_input)
        
        ai_pilot.update_log(ai_input,ai_steer,level_state)
        
        return ai_steer
    
    def get_level_state(self,walker,blocks,finish,frames_left):
        
        # check if game is lost - collision with blocks
        blocks_hit = pg.sprite.spritecollide(walker,blocks,False)
            
        if len(blocks_hit) != 0 or frames_left <= 0:
            return LOST
        
        # check if game is won - collision with finish
        finish_reached = pg.sprite.spritecollide(walker,finish,False)
            
        if len(finish_reached) != 0:
            return WON
        
        return CONTINUE
        
    def start(self,
              ai_pilot = None,
              history_length = 10,
              max_sec = 20):
        
        # get max frames
        frames_left = int(self.fps * max_sec)
        print(frames_left)
        
        # initialize raw level feature history
        raw_level_history = []
        
        # create screen
        self.main_screen = pg.display.set_mode(self.window_size)
        self.main_screen.fill(WHITE)
        
        # get sprite groups
        all_sprites,blocks, finish, walker = self.get_level_sprites(self.levels[0])
        
        # draw all sprites
        all_sprites.draw(self.main_screen)
        pg.display.flip()
        
        # --- game loop: one execution = one frame update
        while True:
            # update frame counter
            frames_left -= 1
            
            # --- append to raw state history and truncate
            raw_level_history.append(pg.surfarray.array3d(self.main_screen))
            raw_level_history = raw_level_history[:history_length]
            
            # --- check for game end criteria
            level_state = self.get_level_state(walker,blocks,finish,frames_left)
            
            # --- check for manual closing of window
            manual_close = False
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    manual_close = True
            
            # --- get steer for walker
            # get player steer if needed
            if ai_pilot == None:
                steer = self.get_player_steer()
            # get ai steer if needed
            elif ai_pilot != None:
                steer = self.get_ai_steer_and_log(ai_pilot,raw_level_history,level_state)
            
            # --- quit if needed: break out of game loop in case of manual quit, level win or level loss
            if manual_close or level_state in (WON,LOST):
                break
            elif level_state == CONTINUE:
                pass
            
            # --- update walker sprite
            walker.update(steer)
            
            # --- redraw screen
            self.main_screen.fill(WHITE)
            all_sprites.draw(self.main_screen)
            pg.display.flip()
            
            # control speed
            self.clock.tick(self.fps)
                
        # quit game
        pg.quit()
        #sys.exit()
        
        return ai_pilot
    
class AI_Walker(object):
    '''Wrapper class to pass to Walk_With_AI that converts raw level state history
    to model inputs and uses these inputs to create an actual steer.'''
    
    def __init__(self,model,feature_converter = None):
        
        # attach specified feature converter;
        if feature_converter == None:
            self.create_input = conv_featurize_latest_frame
        else:
            self.create_input = feature_converter
            
        # attach prediction function that calculates ai steer
        self.create_output = model.predict
        
        # initialize the ai pilot's log to be able to access the history created
        self.log = {'X':[],'y':[], 'reinforce_coeff':None}
        
    def update_log(self,ai_input,ai_steer,level_state):
        
        # update log with current input
        self.log['X'].append(ai_input)
        
        # update log with current steer
        self.log['y'].append(ai_steer)
        
        # populate reinforcement coefficient if appropriate - might need to revisit these
        if level_state == WON:
            self.log['reinforce_coeff'] = 1
        elif level_state == LOST:
            self.log['reinforce_coeff'] = -1
        
def conv_featurize_latest_frame(history_list):
    '''Helper function that takes the latest element of the history list, and formats that
    3-dim array into a 4-dim tensor that can be passed as an input for a convolutional neural
    net as defined in the diy library.'''
    
    assert len(history_list) > 0
    
    # get 3-dim array representation of latest frame
    current_frame = history_list[-1]
    
    # rearrange 3 dimensions into channel, widht, height
    current_frame_rearr = np.transpose(current_frame,(2,0,1))
    
    # add batch size dimension as first dimension
    input_frame = np.expand_dims(current_frame_rearr,axis=0)
    
    return input_frame
    
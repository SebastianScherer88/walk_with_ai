# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:21:09 2019

Game class where character has to navigate obstacle course from designated start
to designated finish. Can be controlled by player or by specified AI.

@author: bettmensch
"""

import sys
import pygame as pg
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
        
        # create marker sprite group
        markers = pg.sprite.Group()
        Block(all_sprites,markers,**level['start'])
        Block(all_sprites,markers,**level['finish'])
        
        # create player sprite group
        walker = pg.sprite.Group()
        walker = Walker(all_sprites,
                        image_surface = self.walker_image,
                        pos_x = level['start']['x'],
                        pos_y = level['start']['y'],
                        theta = 0)
        
        return all_sprites,blocks, markers, walker
    
    def get_player_steer(self):
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return QUIT
            
        keys_pressed = pg.key.get_pressed()
        #print(keys_pressed)
        
        if (keys_pressed[pg.K_LEFT] and keys_pressed[pg.K_RIGHT]) or (not keys_pressed[pg.K_LEFT] and not keys_pressed[pg.K_RIGHT]):
            steer = NONE
        elif (keys_pressed[pg.K_LEFT] and not keys_pressed[pg.K_RIGHT]):
            steer = LEFT
        elif (not keys_pressed[pg.K_LEFT] and keys_pressed[pg.K_RIGHT]):
            steer = RIGHT
                         
        return steer
        
    def start(self,
              ai_pilot = None):
        
        # --- replay loop: one iteration = one game
        while True:
            # create screen
            self.main_screen = pg.display.set_mode(self.window_size)
            self.main_screen.fill(WHITE)
            
            # get sprite groups
            all_sprites,blocks, markers, walker = self.get_level_sprites(self.levels[0])
            
            # draw all sprites
            all_sprites.draw(self.main_screen)
            pg.display.flip()
            
            # --- game loop: one execution = one frame update
            while True:
                # --- get steer for walker
                # get player steer if needed
                if ai_pilot == None:
                    steer = self.get_player_steer()
                # get ai steer if needed
                elif ai_pilot != None:
                    pass
                
                # --- quit if needed: break out of game loop
                if steer == QUIT:
                    break
                
                # --- update walker sprite
                walker.update(steer)
                
                # --- redraw screen
                self.main_screen.fill(WHITE)
                all_sprites.draw(self.main_screen)
                pg.display.flip()
                
                # control speed
                self.clock.tick(self.fps)
                
            # --- quit if needed: break out of replay loop
            if steer == QUIT:
                break
                
        # quit game
        pg.quit()
        sys.exit()
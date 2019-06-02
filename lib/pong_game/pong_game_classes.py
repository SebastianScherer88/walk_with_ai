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

class Paddle(Sprite):
    
    def __init__(self,
                 *groups,
                 pos_x = 0,
                 pos_y = 0,
                 width = PADDLE_WIDTH,
                 length = PADDLE_LENGTH,
                 speed = PADDLE_SPEED,
                 color = WHITE,
                 window_height = WINDOW_SIZE[1]):
        
        # create base sprite instance
        Sprite.__init__(self,
                        *groups)
        
        # create visual attributes
        self.image = pg.Surface((width,length))
        self.image.set_colorkey(BLACK)
        self.rect = self.image.fill(WHITE)
        self.rect.topleft = (pos_x,pos_y)
        
        # create positional attributes
        self.x = vec(pos_x,pos_y)
        self.v_up = vec(0,-speed)
        self.window_height = window_height
        
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
        
        if self.x.y < 0:
            self.x.y = 0
        elif self.x.y > self.window_height - self.rect.height:
            self.x.y = self.window_height - self.rect.height
        
        self.rect.topleft = (int(self.x.x),int(self.x.y))
        
class Ball(Sprite):
    
    def __init__(self,
                 *groups,
                 pos_x = 0,
                 pos_y = 0,
                 v_x = np.sqrt(2),
                 v_y= np.sqrt(2),
                 radius = BALL_RADIUS,
                 color = WHITE,
                 speed = BALL_SPEED,
                 background = BLACK,
                 window_size = WINDOW_SIZE):
        
        # create base sprite instance
        Sprite.__init__(self,
                        *groups)
        
        self.window_size = WINDOW_SIZE
        
        # create visual attributes
        self.image = pg.Surface((2*radius,2*radius))
        self.rect = self.image.fill(background)
        self.rect.topleft = (pos_x,pos_y)
        
        # add circle to image, make edges transparent
        pg.draw.circle(self.image,color,(radius,radius),radius)
        self.image.set_colorkey(background)
        
        # create positional attributes
        self.speed = speed
        self.x = vec(pos_x,pos_y)
        
        self.v = vec(v_x,v_y)
        self.v = speed * self.v.normalize()
        
    def update(self):
        '''Only checks for out of bounds w.r.t height. The collisions with
        paddles are handled in separate method.'''
        
        # if too high, bring back down and bounce
        if self.x.y < 0:
            self.x.y = 0
            self.v.y = -self.v.y
            # or if too low, bring back up and bounce
        elif self.x.y > self.window_size[1] - self.rect.height:
            self.x.y = self.window_size[1] - self.rect.height
            self.v.y = -self.v.y
        
        # update position
        self.x += self.v
        self.rect.topleft = (int(self.x.x),int(self.x.y))
        
    def bounce_off_paddle(self,
                          paddle,
                          min_angle_factor = MIN_BOUNCE_ANGLE_FACTOR):
        '''Check for collisions with paddle and bounce of in an appropriate manner.
        The outgoing bounce angle is determined by the collision point's relative
        position to the bouncing paddle's center, while speed is kept constant.'''
        
        # only bounce off if ball isnt unreasonable past paddle surface
        diff_of_centers = self.rect.centerx - paddle.rect.centerx
        
        if (self.x.x > self.window_size[0] / 2 and diff_of_centers > 0) or (self.x.x < self.window_size[0] / 2 and diff_of_centers < 0):
            return
        
        # get outbound angle
        d_norm = min_angle_factor * (self.rect.centery - paddle.rect.centery) / (paddle.rect.height + self.rect.height) # normalize relative position to [-1,1]
        
        if d_norm < 0:
            d_norm = max(-1,d_norm)
        elif d_norm > 0:
            d_norm = min(1,d_norm)
        
        #print('Cos of outgoing angle:',d_norm)
        
        # adjust velocity ofr horizontal & vertical bounce
        sign_x = np.sign(self.v.x)
        
        self.v.y = self.speed * d_norm
        self.v.x = sign_x * (-1) * self.speed * np.sin(np.arccos(d_norm))
        
        # adjust position to edge of paddle
        if self.x.x > self.window_size[0] / 2:
            self.x.x = paddle.rect.left - self.rect.width
        elif self.x.x < self.window_size[0] / 2:
            self.x.x = paddle.rect.right
            
        self.rect.topleft = self.x
        
class Pong_with_AI(object):
    '''Pong game class. Starts a new game of pong with player controls or AI pilot.'''
    
    def __init__(self,
                 frames_per_second = FRAMES_PER_SECOND,
                 window_size = WINDOW_SIZE,
                 paddle_inset_ratio = PADDLE_INSET_RATIO,
                 colors = {'player':WHITE,
                           'opponent':WHITE,
                           'ball':WHITE,
                           'background':BLACK},
                sizes = {'paddle_width':PADDLE_WIDTH,
                         'paddle_length':PADDLE_LENGTH,
                         'ball_radius':BALL_RADIUS},
                params = {'paddle_speed':PADDLE_SPEED,
                          'ball_speed':BALL_SPEED,
                          'paddle_inset_ratio':PADDLE_INSET_RATIO}):
        
        # initialize pygame window
        pg.init()
        
        # get game clock
        self.clock = pg.time.Clock()
        self.fps = frames_per_second
        
        # some game params
        self.window_size = window_size
        self.paddle_insets = {'opponent':int(self.window_size[0] * paddle_inset_ratio),
                              'player':int(self.window_size[0] * (1 - paddle_inset_ratio) - sizes['paddle_width'])}
        self.colors = colors
        self.sizes = sizes
        self.params = params
        
    def get_level_sprites(self):
        '''Creates and returns the paddle sprites and the ball sprite.'''
        
        # create sprite groups
        all_sprites = pg.sprite.Group()
        paddles = pg.sprite.Group()
        
        # create paddles
        paddle_y = int((self.window_size[1] - self.sizes['paddle_length']) / 2)
        
        opponent_paddle = Paddle(all_sprites, paddles,
                                 pos_x = self.paddle_insets['opponent'],
                                 pos_y = paddle_y,
                                 width = self.sizes['paddle_width'],
                                 length = self.sizes['paddle_length'],
                                 speed = self.params['paddle_speed'],
                                 color = self.colors['opponent'],
                                 window_height = self.window_size[1])
        
        player_paddle = Paddle(all_sprites, paddles,
                               pos_x = self.paddle_insets['player'],
                               pos_y = paddle_y,
                               width = self.sizes['paddle_width'],
                               length = self.sizes['paddle_length'],
                               speed = self.params['paddle_speed'],
                               color = self.colors['player'],
                               window_height = self.window_size[1])
        
        # create ball
        ball_x = int((self.window_size[0] - self.sizes['ball_radius']) / 2)
        ball_y = int((self.window_size[1] - self.sizes['ball_radius']) / 2)
        ball_angle = np.random.uniform(-BALL_INITIAL_MAX_ANGLE,BALL_INITIAL_MAX_ANGLE)
        ball_vy = np.sin(ball_angle)
        ball_vx = np.random.choice([1,-1]) * np.cos(ball_angle)
        
        #print(BALL_INITIAL_MAX_ANGLE)
        #print("initial ball velocity:", ball_vx,ball_vy)
        
        ball = Ball(all_sprites,
                    pos_x = ball_x,
                    pos_y = ball_y,
                    v_x = ball_vx,
                    v_y= ball_vy,
                    radius = self.sizes['ball_radius'],
                    color = self.colors['ball'],
                    speed = self.params['ball_speed'],
                    background = self.colors['background'],
                    window_size = self.window_size)
        
        return all_sprites, paddles, opponent_paddle, player_paddle, ball
    
    def get_player_steer(self):
                    
        keys_pressed = pg.key.get_pressed()
        
        # check for horizontal movement
        if (keys_pressed[pg.K_DOWN] and not keys_pressed[pg.K_UP]):
            steer = DOWN
        elif (keys_pressed[pg.K_UP] and not keys_pressed[pg.K_DOWN]):
            steer = UP
        else:
            steer = NONE
                         
        return steer
    
    def get_ai_steer_and_log(self,ai_pilot,level_history,level_state):
            
        if len(level_history) >= 2:
            # create input to ai pilot from raw level states
            ai_input = ai_pilot.create_input(level_history)
            
            ai_steer = ai_pilot.create_output(ai_input)
            
            ai_pilot.update_log(ai_input,ai_steer,level_state)
        else:
            ai_steer = NONE
        
        return ai_steer
    
    def get_opponent_steer(self,ball,opponent_paddle,stat_margin = OPPONENT_STATIONARY_MARGIN):
        
        if opponent_paddle.rect.centery - ball.rect.centery > (stat_margin * opponent_paddle.rect.height):
            steer = UP
        elif opponent_paddle.rect.centery - ball.rect.centery < -(stat_margin * opponent_paddle.rect.height):
            steer = DOWN
        else:
            steer = NONE
           
        #print("opponent center - ball center:",opponent_paddle.rect.centery - ball.rect.centery)
        #print("opponent steer:",steer)
            
        return steer
    
    def get_level_state(self,ball,frames_left):
        
        ball_pos = ball.rect.centerx

        # check if game is lost - ball is on the right, beyond player paddle            
        if ball_pos > self.window_size[0]:
            return LOST
        # check if game is lost - ball is on the left, beyond opponent paddle
        elif ball_pos < 0: 
            return WON
        # check if game is lost - time is up
        if frames_left <= 0:
            return TIMEDOUT
        
        return CONTINUE
        
    def start(self,
              ai_pilot = None,
              history_length = 10,
              max_frames = 200,
              visual = True,
              level_history_type = 'conv'):
        
        assert level_history_type in ('conv','mlp')
        
        # get max frames
        frames_left = max_frames
        #print(frames_left)
        
        # initialize level feature history
        level_history = []
        
        # create screen according to whether video is available or not
        if visual:
            self.main_screen = pg.display.set_mode(self.window_size)
        else:
            self.main_screen = pg.Surface(self.window_size)
            
        self.main_screen.fill(self.colors['background'])
        
        # get sprite groups
        all_sprites, paddles, opponent_paddle, player_paddle, ball = self.get_level_sprites()
        
        # draw all sprites
        all_sprites.draw(self.main_screen)
        
        if visual:
            pg.display.flip()
        
        # --- game loop: one execution = one frame update
        while True:
            # update frame counter
            frames_left -= 1
            
            # --- append to state history and truncate
            if level_history_type == 'conv':
                level_history.append(pg.surfarray.array3d(self.main_screen))
            elif level_history_type == 'mlp':
                y_array = np.array([opponent_paddle.x.y,player_paddle.x.y])
                positional_level_snapshot = np.concatenate([y_array,ball.x,ball.v]).reshape(1,-1)
                level_history.append(positional_level_snapshot)
            
            level_history = level_history[:history_length]
            
            # --- check for game end criteria
            level_state = self.get_level_state(ball,frames_left)
            #print('Level state (from within game):',level_state)
            
            # --- check for manual closing of window
            manual_close = False
            
            if visual:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        manual_close = True
            
            # --- get steer for paddles
            # get player steer if needed
            if ai_pilot == None:
                player_steer = self.get_player_steer()
            # get ai steer if needed
            elif ai_pilot != None:
                    
                player_steer = self.get_ai_steer_and_log(ai_pilot,
                                                         level_history,
                                                         level_state)
                
            opponent_steer = self.get_opponent_steer(ball,opponent_paddle)
            
            # --- quit if needed: break out of game loop in case of manual quit, level win or level loss
            if manual_close or level_state in (WON,LOST,TIMEDOUT):
                break
            elif level_state == CONTINUE:
                pass
            
            # --- update paddles & ball
            player_paddle.update(player_steer)
            opponent_paddle.update(opponent_steer)
            ball.update()
            
            # --- check for ball vs paddle bounces
            bounce_paddles = pg.sprite.spritecollide(ball, paddles, False, pg.sprite.collide_mask)

            if len(bounce_paddles) != 0:
                ball.bounce_off_paddle(bounce_paddles[0])
            
            # --- redraw screen
            self.main_screen.fill(self.colors['background'])
            all_sprites.draw(self.main_screen)
                
            if visual:
                pg.display.flip()
            
            # control speed
            self.clock.tick(self.fps)
                
        # quit game
        pg.quit()
        #sys.exit()
        
        return ai_pilot
    
class AI_Pong(object):
    '''Wrapper class to pass to Pong_With_AI that converts raw level state history
    to model inputs and uses these inputs to create an actual steer.'''
    
    def __init__(self,model,net_type='conv',level_state_rewards = REWARDS_MAP):
        
        # attach specified feature converter
        assert (net_type in ('conv','mlp'))
        if net_type == 'conv':
            self.create_input = conv_featurize_difference_last_two_frames
        elif net_type == 'mlp':
            self.create_input = normalize_positional_stats
            
        # attach model
        self.model = model
        
        # attach rewards map
        self.rewards = level_state_rewards
        
        # initialize the ai pilot's log to be able to access the history created
        self.log = {'X':[],'y':[], 'reinforce_coeff':None}
        
    def create_output(self,feature_state):
        # get class conditional distribution = classification network output without argmaxing
        cond_class_distr = self.model.predict(feature_state,distribution = True)
        
        # sample action = move from class conditional distribution
        #print(cond_class_distr)
        #print(type(cond_class_distr))
        
        #print(self.model.classes_ordered)
        #print(type(self.model.classes_ordered))
        
        output = self.model.classes_ordered[np.argmax(np.random.multinomial(1,cond_class_distr[0]))]
        
        #print(output)
        
        return output
        
    def update_log(self,ai_input,ai_steer,level_state,n_loss_cause = 100):
        
        # update log with current input
        self.log['X'].append(ai_input)
        
        #print("ai steer:",ai_steer)
        
        # update log with current steer
        self.log['y'].append(ai_steer)
        
        # populate reinforcement coefficient if appropriate - might need to revisit these
        if level_state in (WON,LOST,TIMEDOUT):
            self.log['reinforce_coeff'] = self.rewards[level_state]

            # cut down losing trajectory to most recent mistakes more likely to have caused the loss            
            if level_state == LOST:
                self.log['X'] = self.log['X'][-n_loss_cause:]
                self.log['y'] = self.log['y'][-n_loss_cause:]
                
        #print(len(self.log['X']))
        #print(len(self.log['y']))
        #print('Level state:',level_state)
        #print(self.log['reinforce_coeff'])
            
        
def conv_featurize_difference_last_two_frames(history_list):
    '''Helper function that takes the difference of the two latest elements of the history list, and formats that
    3-dim array into a 4-dim tensor that can be passed as an input for a convolutional neural
    net as defined in the diy library.'''
    
    assert len(history_list) >= 2
    
    # get 3-dim array representation of latest frame
    difference_frame = history_list[-1] - history_list[-2]
    
    # rearrange 3 dimensions into channel, widht, height
    difference_frame_rearr = np.transpose(difference_frame,(2,0,1))
    
    # add batch size dimension as first dimension
    input_frame = np.expand_dims(difference_frame_rearr,axis=0)
    
    return input_frame

def normalize_positional_stats(history_list,
                               ball_speed = BALL_SPEED,
                               window_width = WINDOW_SIZE[0],
                               window_height = WINDOW_SIZE[1]):
    '''Helper function that takes a numpy array of the form
    [opponent_y,player_y,ball_x,ball_y,ball_vx,ball_vy],
    normalizes positional parameters by window size and velocity parameters
    by ball's speed. Returns a numpy array of the same dimension.'''
    
    assert len(history_list) >= 1
    
    # get most recent data point
    current_positional_state = history_list[-1]
    
    # scale the y coordinates to [-1,1]
    y_coord_pos = np.array([True,True,False,True,False,False])
    current_positional_state[0,y_coord_pos] = (current_positional_state[0,y_coord_pos] - 0.5 * window_height) / window_height
    
    # scale the x coordinates to [-1,1]
    x_coord_pos = np.array([False,False,True,False,False,False])
    current_positional_state[0,x_coord_pos] = (current_positional_state[0,x_coord_pos] - 0.5 * window_width) / window_width
    
    # scale the ball velocity coordinates to [-1,1]
    ball_v_coord_pos = np.array([False,False,False,False,True,True])
    current_positional_state[0,ball_v_coord_pos] = (current_positional_state[0,ball_v_coord_pos] - 0.5 * ball_speed) / ball_speed
    
    return current_positional_state
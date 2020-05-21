#!/usr/bin/env python
# coding: utf-8

######################
####  PARAMETERS. ####
######################
do_load_model = False       ### Load model or create from scratch?
model_name = 'player_v3.h5'
fixed_start = True      ### Always start from same config?
run = 1			 ### Index for saving figures
epochs = 500             ### Number of cycles
max_steps = 200 	 ### Max steps per epoch
world_size = 20         ### Linear dimensions of the (squared) world
see_size = 20            ### DO NOT CHANGE. Dimension of the observable world.
                         ### If you change it: you have to change the DRL model input size
group_size = 4           ### DO NOT CHANGE. Linear dimensions of the (suqared) community
                         ### If you change it: you have to change the DRL model output size
filling = 0.40	         ### How "full" the starting world is

IsWorldFuzzy = False    ### "Fuzzy" world means that cells have a random chance of switching
p_fuzzy = 1.0/world_size/world_size     ### Note that approx. P(1 switch) = world_size*world_size*p_fuzzy
                       ### If world_size ~ 100, p_fuzzy should be ~ 0.01 ~ to get 100 events (over 10 thousand squares)

alive = 1              ### Def. 1 alive and 0 dead 
dead = 0               ###
seed = 2	       ### Fixed start trial 

from IPython import display
import gc
import numpy as np
import matplotlib as mtplt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
mtplt.use('Qt5Agg')
import matplotlib.animation as animation
import time
import copy
from matplotlib import rc

import random
random.seed(seed)


### IMPORTING KERAS MODULES!
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
print(sys.path)
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
import gym
gol_env = gym.make('gym_gol:gol-v0')
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

if do_load_model:
    print('Loading model :')
    t0 = time.time()
    player = load_model(model_name)
    t1 = time.time()
    print('Model loaded in: ', t1-t0)
else:
   player_input = Input(shape=(1,see_size, see_size))
   ## 20x20x1
   x = Conv2D(16, (2, 2), activation='relu', padding='same')(player_input)
   ## 20x20x16
   x = MaxPooling2D((2, 2), padding='same')(x)
   ## 10x10x16
   x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
   ## 10x10x16
   x = MaxPooling2D((2, 2), padding='same')(x)
   ## 5x5x16
   x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
   ## 5x5x16
   x = MaxPooling2D((5, 5), padding='same', name='encoder')(x)
   ## 1x1x16
   x = Flatten()(x)
   x = Dense(16, activation = 'relu')(x)  ## Dense layer with 16 nodes
   player_output = Dense(16, activation = 'relu')(x) ## Final layer with 16 output nodes
   model = Model(inputs=player_input, outputs=player_output)
   print(model.summary())   ## DEBUG

gol_env.seed(seed)
world = gol_env.reset()
print(world)
y = 0.95
eps = 0.5
decay_factor = 0.999

memory = SequentialMemory(limit=10000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)
player = DQNAgent(model=model, nb_actions=16, memory=memory, nb_steps_warmup=50,
               target_model_update=1e-2, policy=policy)

player.compile(Adam(lr=1e-3),metrics=['mae'])
player.fit(gol_env, nb_steps=5000, visualize=False, verbose=1)
player.test(gol_env, nb_episodes=5, visualize=False, verbose=1)

player.save(model_name)
exit()

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
from keras.callbacks import TensorBoard
import gym
gol_env = gym.make('gym_gol:gol-v0')
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import keras.backend as K_backend

print(gol_env.step(10))

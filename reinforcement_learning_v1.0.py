#!/usr/bin/env python

# coding: utf-8


######################
####  PARAMETERS. ####
######################
epochs = 100            ### Number of cycles
max_steps = 500		 ### Max steps per epoch
world_size = 200         ### Linear dimensions of the (squared) world
see_size = 20            ### DO NOT CHANGE. Dimension of the observable world.
                         ### If you change it: you have to change the DRL model input size
group_size = 6           ### DO NOT CHANGE. Linear dimensions of the (suqared) community
                         ### If you change it: you have to change the DRL model output size
filling = 0.50	         ### How "full" the starting world is

IsWorldFuzzy = False    ### "Fuzzy" world means that cells have a random chance of switching
p_fuzzy = 1.0/world_size/world_size     ### Note that approx. P(1 switch) = world_size*world_size*p_fuzzy
                       ### If world_size ~ 100, p_fuzzy should be ~ 0.01 ~ to get 100 events (over 10 thousand squares)

alive = 1              ### Def. 1 alive and 0 dead 
dead = 0               ###


# In[2]:

from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import copy
from matplotlib import rc

import random
start = time.time()

from gameofsurvival_libraries import init_grid, grid_to_list, grid_to_set, list_to_grid, set_to_grid
from gameofsurvival_libraries import get_neighbours, apply_rules, fuzzy_rules, calculate_action, time_step
from gameofsurvival_libraries import alive_cells


# In[3]:


### Generate world and group at random
world = init_grid(world_size)
group = init_grid(group_size)

### PLOT INITIAL STATE
print("###STARTING WORLD CONFIGURATION###")
#plt.imshow(world)
#plt.show()
#time.sleep(1)
#plt.close()

### IMPORTING KERAS MODULES!
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
print(sys.path)
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# In[5]:


player = Sequential()
player.add(Input(shape=(see_size, see_size,1)))
## 20x20x1
player.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
## 20x20x16
player.add(MaxPooling2D((2, 2), padding='same'))
## 10x10x16
player.add(Conv2D(36, (3, 3), activation='relu', padding='same'))
## 10x10x36
player.add(MaxPooling2D((2, 2), padding='same'))
## 5x5x36
player.add(Conv2D(36, (2, 2), activation='relu', padding='same'))
## 5x5x36
player.add(MaxPooling2D((6, 6), padding='same', name='encoder'))
## 1x1x36
player.add(Flatten())
player.compile(loss='mse', optimizer='adam', metrics=['mae'])

y = 0.95
eps = 0.5
decay_factor = 0.999
r_avg_list = []
sight_index_1 = int((world_size-see_size)/2)
sight_index_2 = int((world_size+see_size)/2)

fig, ax = plt.subplots(figsize=(14, 12))
fig.canvas.draw()
img = ax.imshow(world, interpolation='none')
action = np.zeros(group_size*group_size)

for i in range(epochs):
    world = init_grid(world_size)     ### Initialize world
    alive_cells = grid_to_set(world)
    eps *= decay_factor               ### Decay probability of random action
    print("Epoch", i+1,  "of ", epochs)
    done = False
    r_sum = 0
    done = 0
    for j in range(max_steps):
        if np.random.random() < eps:
            action = np.argmax(np.random.random(group_size*group_size))
        else:
            action = np.argmax(player.predict(np.reshape(world[sight_index_1:sight_index_2,
                                                               sight_index_1:sight_index_2],
                                                         (1,see_size,see_size,1)).astype(float)))
        ### FIRST STEP: the agent performs an action on the world
        world = set_to_grid(alive_cells, world_size)
        next_world = calculate_action(action, world, world_size, group_size)
        alive_cells = grid_to_set(next_world)
        ### SECOND STEP: time evolution (including the action of the agent)
        ### RETURNS: the reward, the status of the world, and the variable "Done"
        reward, next_world, done = time_step(alive_cells, world_size, IsWorldFuzzy, p_fuzzy)
        ### UPDATE learner
        q_step = reward + y * np.max(player.predict(np.reshape(next_world[sight_index_1:sight_index_2,
                                                               sight_index_1:sight_index_2],
                                                         (1,see_size,see_size,1)).astype(float)))
        q_a_vec = player.predict(np.reshape(world[sight_index_1:sight_index_2,
                                                               sight_index_1:sight_index_2],
                                                         (1,see_size,see_size,1)).astype(float))[0]
        q_a_vec[action] = q_step
        player.fit(np.reshape(world[sight_index_1:sight_index_2,
                        sight_index_1:sight_index_2],
                    (1,see_size,see_size,1)).astype(float), 
           q_a_vec.reshape(-1,36), batch_size=1, epochs=1, verbose=0)
        r_sum += reward
        ### THIRD STEP: save "old" configuration
        world = next_world
        alive_cells = grid_to_set(world)
        ### INFORMATIVE MESSAGE
        if (j%(int(max_steps/4))==0):
            print(float(j)/max_steps*100, "% of steps done")
        ### PLOT WORLD STATUS
        if i >= 0:
            alive_cells = grid_to_set(world)
##            img.set_data(world)
##            display.clear_output(wait=True)
##            display.display(fig)
##            time.sleep(0.00000001)
        if done == 1:
            break
    r_avg_list.append(r_sum / epochs)

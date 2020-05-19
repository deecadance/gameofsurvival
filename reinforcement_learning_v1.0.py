#!/usr/bin/env python
# coding: utf-8

######################
####  PARAMETERS. ####
######################
run = 1			 ### Index for saving figures
epochs = 100             ### Number of cycles
max_steps = 400		 ### Max steps per epoch
world_size = 100         ### Linear dimensions of the (squared) world
see_size = 20            ### DO NOT CHANGE. Dimension of the observable world.
                         ### If you change it: you have to change the DRL model input size
group_size = 6           ### DO NOT CHANGE. Linear dimensions of the (suqared) community
                         ### If you change it: you have to change the DRL model output size
filling = 0.35	         ### How "full" the starting world is

IsWorldFuzzy = False    ### "Fuzzy" world means that cells have a random chance of switching
p_fuzzy = 1.0/world_size/world_size     ### Note that approx. P(1 switch) = world_size*world_size*p_fuzzy
                       ### If world_size ~ 100, p_fuzzy should be ~ 0.01 ~ to get 100 events (over 10 thousand squares)

alive = 1              ### Def. 1 alive and 0 dead 
dead = 0               ###

from IPython import display
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
start = time.time()

from gameofsurvival_libraries import init_grid, grid_to_list, grid_to_set, list_to_grid, set_to_grid
from gameofsurvival_libraries import get_neighbours, apply_rules, fuzzy_rules, calculate_action, time_step
from gameofsurvival_libraries import alive_cells, counter


### Generate world and group at random
world = init_grid(world_size, filling)
group = init_grid(group_size, filling)

### PLOT INITIAL STATE
print("###STARTING WORLD CONFIGURATION###")
#plt.imshow(world)
#plt.show(block=False)
#time.sleep(2)
#plt.close()
#plt.ioff()

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
group_index_1 = int((world_size-group_size)/2)
group_index_2 = int((world_size+group_size)/2)

action = np.zeros(group_size*group_size)

for i in range(epochs):
    world = init_grid(world_size, filling)     ### Initialize world
    alive_cells = grid_to_set(world)
    eps *= decay_factor               ### Decay probability of random action
    print("Epoch", i+1,  "of ", epochs)
    done = False
    r_sum = 0
    done = 0
    time = []
    group_alive_percent = []
    control_alive_percent = []
    world_alive_percent = []
    reward_history = []
    max_t_list = []
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
        reward, next_world, done = time_step(alive_cells, world_size, group_size, IsWorldFuzzy, p_fuzzy)
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
        ### SAVE DATA:
        group_alive_cells, control_alive_cells, world_alive_cells = counter(world, world_size, group_size)
        group_alive_percent.append(group_alive_cells/group_size/group_size*100)
        control_alive_percent.append(control_alive_cells/group_size/group_size*100)
        world_alive_percent.append((world_alive_cells-group_alive_cells-control_alive_cells)/(world_size*world_size-2*group_size*group_size)*100)
        reward_history.append(reward)
        time.append(j)
        ### INFORMATIVE MESSAGE
        if (j%(int(max_steps/4))==0):
            print(float(j)/max_steps*100, "% of steps done")
        ### PLOT WORLD STATUS
        if (i == epochs-1):
            print("Saving gameplay for the last epoch")
            alive_cells = grid_to_set(world)
            plt.ioff()
            fig, ax = plt.subplots()
            im = ax.imshow(world)
            ax.set_xlim(sight_index_1-5,sight_index_2+5)
            ax.set_ylim(sight_index_1-5,sight_index_2+5)
            rect1 = patches.Rectangle((group_index_1,group_index_1),group_size,group_size,linewidth=1,edgecolor='r',facecolor='none')
            rect2 = patches.Rectangle((sight_index_1,sight_index_1),see_size,see_size,linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            x = int(action/group_size) + group_index_1
            y = action%group_size + group_index_1
            ax.scatter(x,y,c='cyan',marker='s',s=16)
            fig.colorbar(im)
            name = './run_' + str(run) + '_epoch_' + str(i) + '_t_' + str(j) + '.png'
            plt.savefig(name)
            plt.close(fig)
        if done == 1:
            break
    max_t_list.append(i)
    r_avg_list.append(r_sum/epochs)
    name = 'epoch_' + str(i) + '_days_' + str(j) + '_metrics.txt'
    time_nparray = np.asarray(time)
    wap_nparray = np.asarray(world_alive_percent)
    gap_nparray = np.asarray(group_alive_percent)
    cap_nparray = np.asarray(control_alive_percent)
    rew_nparray = np.asarray(reward_history)
    np.savetxt(name,np.column_stack([time_nparray,wap_nparray,gap_nparray,cap_nparray, rew_nparray]),
header='step, world, group,control, reward')

max_t_nparray = np.asarray(max_t_list)
np.savetxt('game_length.txt',np.transpose(max_t_nparray))
player.save('player_v1.h5')

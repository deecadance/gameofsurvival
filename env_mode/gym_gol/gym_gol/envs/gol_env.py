#!/usr/bin/env python

import gym
import logging.config
import pkg_resources
import cfg_load

from gym import error, spaces, utils

from gym_gol.envs.gol_view_2d import GolView2D
from gym.utils import seeding
import numpy as np
import time
import random

path = "config.yaml"
filepath = pkg_resources.resource_filename("gym_gol", path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config["LOGGING"])


class GolEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self,epochs=200,world_size=50,group_size=4,see_size=20,filling=0.50,fixed_start=True,seed=42, max_iter=5000):
    self.__version__ = "0.0.1"
    logging.info(f"GolEnv - Version {self.__version__}")
    ### RUN VARIABLES         
    self.curr_step = -1             ## Count number of steps
    self.curr_episode = -1
    self.done = False               ## Exit variable
    self.epochs = epochs  			## How many epochs
    self.fixed_start = fixed_start  ## Bool: always start in the same config?
    self.max_iter = max_iter
    ### WORLD VARIABLES
    self.world_size = world_size    ## World size for the world to evolve in
    self.group_size = group_size    ## Group size for the model to act on
    self.see_size = see_size	## Sight size for the model to learn on
    self.filling = filling          ## Filling ratio for initialization
    self.group_index_1 = int((self.world_size-self.group_size)/2)
    self.group_index_2 = int((self.world_size+self.group_size)/2)
    self.sight_index_1 = int((self.world_size-self.see_size)/2)
    self.sight_index_2 = int((self.world_size+self.see_size)/2)	
    self.world = self._init_grid()  ## Constructing World
    self.next_world = self.world    ## World copy
    self.alive_cells = self._grid_to_set(self.world)  ## Transforming world into set represent
    ### AGENT VARIABLES
    self.action_space = \
      spaces.Discrete(self.group_size*self.group_size) ## Discrete set of integers representing actions 
    self.observation_space = \
    spaces.Discrete(self.see_size*self.see_size)
    self.action_episode_memory = []
    self.last_action = 0
    ### VISUAL VARIABLES
    self.visualization = None
  def step(self, action):
      """
      The agent takes a step in the environment.
      Parameters
      action : group_size x group_size array
      Returns
      observation (array) : see_size x see_size array of zeros and one
      reward (float) : Amount of reward achieved by previous action
      done (bool) : True if population inside the group is estinguished
         done being True indicates the episode has terminated. 
      """
      if self.done :
          raise RuntimeError("Episode is done")
      else :
          self.curr_step += 1
          self._take_action(action)   ### ACTION IS TAKEN: changes self.next_world
          self.alive_cells = self._grid_to_set(self.next_world)  ### WORLD EVOLVES: switch to alive_cells rep.
          self._apply_rules()					 ### WORLD EVOLVES: apply rules to alive_cells
          self.next_world = self._set_to_grid(self.alive_cells, self.world_size)  ### WORLD EVOLVES: back to grid rep.
          reward = self._get_reward()
          observation = self.next_world[self.sight_index_1:self.sight_index_2,self.sight_index_1:self.sight_index_2]
          self.world = self.next_world
          return observation, reward, self.done, {"Filling": self.filling, "Action": action, "Done": self.done}

  def _take_action(self, action): ### Here the action is taken, and the world is evolved.
    ### Add current action to memory ###
    self.last_action = action
    self.action_episode_memory[self.curr_episode].append(action)
    x = int(action/self.group_size) + self.group_index_1
    y = action%self.group_size + self.group_index_1
    self.next_world = self.world
    self.next_world[x,y] = (self.world[x,y]+1)%2
         
  def _get_reward(self):
    if not np.any(self.next_world[self.group_index_1:self.group_index_2,self.group_index_1:self.group_index_2]):
      self.done = True
      return -self.group_size*self.group_size*(self.max_iter - self.curr_step)/100
    elif (self.curr_step >= self.max_iter) :
      self.done = True
      print("Survived until the end!!")
      return self.max_iter*self.group_size*self.group_size
    else:
      self.done = False
      return np.sum(self.next_world[self.group_index_1:self.group_index_2,self.group_index_1:self.group_index_2])

  def reset(self):
    self.curr_step = -1
    self.curr_episode = +1
    self.action_episode_memory.append([])
    self.done = False
    self.world = self._init_grid() 
    return self.world[self.sight_index_1:self.sight_index_2,self.sight_index_1:self.sight_index_2]
 

  ### Initialize Game of Life World ###
  def _init_grid(self):
    return np.random.choice([0,1], self.world_size*self.world_size, 
          p=[1-self.filling, self.filling]).reshape(self.world_size,self.world_size).astype(int)

  def render(self, mode='human'):
      if mode == 'text' :
          with np.printoptions(threshold=np.inf):
              print("Group at step", self.curr_step)
              print(self.next_world)
      if mode == 'human' :
        if self.visualization == None:
          self.visualization = GolView2D(gol_file_path=None, world_size = self.world_size, group_size = self.group_size, see_size = self.see_size, enable_render = True, world = self.world, sight_index_1 = self.sight_index_1, sight_index_2 = self.sight_index_2, group_index_1 = self.group_index_1, group_index_2 = self.group_index_2, action = self.last_action, curr_episode = self.curr_episode,
curr_step = self.curr_step) 
        if self.curr_step > 0:
          self.visualization.render(self.next_world, self.curr_step, self.curr_episode, self.last_action)
      return
  
  def seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    return

  def _grid_to_list(grid):
    population = np.argwhere(grid).tolist() 
    return population
  
  def _grid_to_set(self, grid):
      population = np.argwhere(grid).tolist()
      population = tuple(map(tuple, population))
      population = set(population)
      return population
  
  def _list_to_grid(population, world_size):
      new_grid = np.zeros((world_size, world_size))
      if not population:
        return new_grid
      else:
        row_indices = population[:,0]
        col_indices = population[:,1]
        new_grid[row_indices,col_indices] = 1
        return new_grid 

  def _set_to_grid(self, population, world_size):
    new_grid = np.zeros((world_size, world_size))
    population = list(population)
	### BUGFIX v0.1: before it would return an error if the list was empty
    if not population:  
      return new_grid
    else:
        population = np.array(population, dtype = int)
        row_indices = population[:,0]
        col_indices = population[:,1]
        new_grid[row_indices,col_indices] = 1
        return new_grid

  def _get_neighbours(self,  element):
    l = []
    l.append( ( (element[0]-1)%self.world_size, (element[1]  )%self.world_size ) )
    l.append( ( (element[0]-1)%self.world_size, (element[1]+1)%self.world_size ) )
    l.append( ( (element[0]-1)%self.world_size, (element[1]-1)%self.world_size ) )
    l.append( ( (element[0]  )%self.world_size, (element[1]+1)%self.world_size ) )
    l.append( ( (element[0]  )%self.world_size, (element[1]-1)%self.world_size ) )
    l.append( ( (element[0]+1)%self.world_size, (element[1]+1)%self.world_size ) )
    l.append( ( (element[0]+1)%self.world_size, (element[1]-1)%self.world_size ) )
    l.append( ( (element[0]+1)%self.world_size, (element[1]  )%self.world_size ) )
##    print("Neighbours: ", l)
    return l

  ## SET OF RULES ON SPARSE SET
  def _apply_rules(self):
      counter = {}
      for cell in self.alive_cells:
          if cell not in counter: ## You don't want to look twice at the same cell
              counter[cell] = 0   ## Initialize counter for alive cells
          neighbours = self._get_neighbours(cell) ## Obtain a LIST containing the coordinates of neighbours
          for n in neighbours:
              if n not in counter: ## Cells not in the counter are currently dead
                  counter[n] = 1   ## Initialize them with 1 (the current neighbour)
              else:                ## Cells already in the counter are alive
                  counter[n] += 1  ## Increment their counter by one
      for c in counter:            ## Now look at the newly created list and apply rules
          if (counter[c] < 2 or counter[c] > 3):
              self.alive_cells.discard(c)
          if (counter[c] == 3):
              self.alive_cells.add(c)   ## Add or discard cells according to rules
      return 
  def close(self):
      pass

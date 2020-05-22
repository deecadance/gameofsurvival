import numpy as np
import matplotlib as mtplt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GolView2D:
    def __init__(self, world, sight_index_1, sight_index_2,
                     group_index_1, group_index_2, action,
                     gol_file_path=None, world_size = 50, group_size = 4,
                     see_size = 20, screen_size = (600,600),
                     has_loops = False, enable_render = True, curr_step = 0, curr_episode = 0):
  
        self.gol_file_path = gol_file_path
        self.world_size = world_size
        self.group_size = group_size
        self.see_size = see_size
        self.screen_size = screen_size
        self.__enable_render = enable_render
        self.curr_step = curr_step       
        self.curr_episode = curr_episode

        self.title = None
        self.world = world  
        self.sight_index_1 = sight_index_1
        self.sight_index_2 = sight_index_2
        self.group_index_1 = group_index_1
        self.group_index_2 = group_index_2
        self.action = action
##        plt.show(block=False)

    def render(self, plot_world, curr_step, curr_episode, action):
        if curr_step > 0:
            rect1 = patches.Rectangle((self.group_index_1-0.5,self.group_index_1-0.5),self.group_size,self.group_size,linewidth=1,edgecolor='r',facecolor='none')
            rect2 = patches.Rectangle((self.sight_index_1-0.5,self.sight_index_1-0.5),self.see_size,self.see_size,linewidth=1,edgecolor='g',facecolor='none')
            x = int(action/self.group_size) + self.group_index_1
            y = action%self.group_size + self.group_index_1
            fig, ax = plt.subplots()
            ax.set_xlim(self.sight_index_1-5,self.sight_index_2+5)
            ax.set_ylim(self.sight_index_1-5,self.sight_index_2+5)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            im = ax.imshow(plot_world)
            ax.scatter(x,y,c='cyan',marker='s',s=16)
##            plt.pause(0.001) 
            name = './run_' + '_epoch_' + str(curr_episode) + '_t_' + str(curr_step) + '.pdf'
            plt.savefig(name)
            plt.close(fig)

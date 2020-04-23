import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
from rose import Car, Map, car_colors
from PIL import Image
import _pickle as pickle
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


main_dir = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))
car_figs = dict()
for color in car_colors:
    car_figs[color] = main_dir + '/rose/cars/' + color + '_car.png'


# animate the files completely
def traces_to_animation(filename):
    # extract out traces from pickle file
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)

    the_map = Map(traces['map_name'])
    t_end = traces['t_end']
    global ax
    fig, ax = plt.subplots()

    # plot out agents and traffic lights
    for t in range(t_end):
        print(t)
        ax.cla()
        agents = traces[t]['agents']
        lights = traces[t]['lights']
        plot_map(the_map)
        plot_cars(agents)
        plot_traffic_lights(lights)
        plot_name = str(t).zfill(5)
        img_name = os.getcwd()+'/imgs/plot_'+plot_name+'.png'
        fig.savefig(img_name)
    animate_images()

def plot_cars(agents):
    for i, agent in enumerate(agents):
        # draw the car with its bubble
        draw_car(agent)

def plot_traffic_lights(traffic_lights):
    for i, light_node in enumerate(traffic_lights):
        #print(light_node)
        color = light_node[2]
        if color == 'red':
            c = 'r'
        if color == 'green':
            c = 'g'
        if color == 'yellow':
            c = 'y'
        rect = patches.Rectangle((light_node[1],light_node[0]), 1,1,linewidth=0,facecolor=c, alpha=0.2)
        ax.add_patch(rect)

def get_map_corners(map):
    grid = map.grid
    x_lo, x_hi = min(list(zip(*grid))[0]), max(list(zip(*grid))[0])
    y_lo, y_hi = min(list(zip(*grid))[1]), max(list(zip(*grid))[1])

    # redefine in coordinate frame
    x_min = y_lo
    x_max = y_hi+1
    y_min = x_lo
    y_max = x_hi+1
    return x_min, x_max, y_min, y_max

# defining a function that plots the map on a figure
def plot_map(map, grid_on=True):
    x_min, x_max, y_min, y_max = get_map_corners(map)
    ax.axis('equal')
    ax.minorticks_on()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # fill in the obstacle regions
    for obs in map.non_drivable_nodes:
        rect = patches.Rectangle((obs[1],obs[0]), 1,1,linewidth=1,facecolor='k', alpha=0.3)
        ax.add_patch(rect)

    plt.gca().invert_yaxis()
    if grid_on:
        ax.grid()
        plt.axis('on')
    else:
        plt.axis('off')

def draw_car(agent_state_tuple):
    # global params
    x, y, theta, v, color, bubble, ag_id = agent_state_tuple
    theta_d = Car.convert_orientation(theta)
    car_fig = Image.open(car_figs[color])
    # need to flip since cars are inverted
    if theta_d == np.pi/2: 
        theta_d = np.pi
    elif theta_d == np.pi:
        theta_d = np.pi/2

    car_fig = car_fig.rotate(theta_d, expand=False)
    offset = 0.1
    ax.imshow(car_fig, zorder=1, interpolation='none', extent=[y+offset, y+1-offset, x+offset, x+1-offset])

    for grid in bubble:
        rect = patches.Rectangle((grid[1],grid[0]),1,1,linewidth=0.5,facecolor='grey', alpha=0.1)
        ax.add_patch(rect)


# to plot a single bubble for the paper figure
def plot_bubble(bubble):
    #global ax, fig
    #fig, ax = plt.subplots()
    #ax.axis('equal')

    for grid in bubble:
        rect = patches.Rectangle((grid[1],grid[0]),1,1,linewidth=0.5,facecolor='grey', alpha=0.5)
        ax.add_patch(rect)

    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # draw the car
    #fig = Image.open(car_figs['green'])
    #agent_state_tuple = (0, 0, 'east', 'green')
    #draw_car(agent_state_tuple)

    #ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #ax.grid(which='both')

    plt.gca().invert_yaxis()
    plt.show()
    #return ax

def make_bubble_figure(bubble_file):
    def plt_car(ax, car_tuple):
        x, y, theta, v, color, bubble, ag_id = car_tuple
        theta_d = Car.convert_orientation(theta)
        car_fig = Image.open(car_figs[color])
        # need to flip since cars are inverted
        if theta_d == np.pi/2: 
            theta_d = np.pi
        elif theta_d == np.pi:
            theta_d = np.pi/2

        car_fig = car_fig.rotate(theta_d, expand=False)
        offset = 0.1
        ax.imshow(car_fig, zorder=1, interpolation='none', extent=[y+offset, y+1-offset, x+offset, x+1-offset])

    with open(bubble_file, 'rb') as pckl_file:
        all_bubbles = pickle.load(pckl_file)

    fig = plt.figure()
    car_tuple = (0, 0, 'east', 0, 'orange', [(0,0)], 0)
        
    for key, bubble in all_bubbles.items():
        ax = fig.add_subplot(1,4,key+1)
        ax.set_title('bubble for v = {}'.format(key), fontdict={'fontsize': 18, 'fontweight': 'medium', 'family':'serif'})
        ax.set_xlim(-5, 11)
        ax.set_ylim(-5, 6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for grid in bubble:
            rect = patches.Rectangle((grid[1],grid[0]),1,1,linewidth=0.25,facecolor='orange', alpha=0.2)
            ax.add_patch(rect)
        plt_car(ax, car_tuple)

    
    plt.show()

def animate_images():
    # Create the frames
    frames = []
    imgs = glob.glob(os.getcwd()+'/imgs/plot_'"*.png")
    imgs.sort()
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(os.getcwd()+'/imgs/' + 'png_to_gif.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=200, loop=3)

if __name__ == '__main__':
    output_dir = os.getcwd()+'/imgs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    traces_file = os.getcwd()+'/saved_traces/debugging_conflict.p'
    traces_to_animation(traces_file)
    animate_images()

    # bubbles figure for the paper
    # for dynamics a:-1,1, v=3
    #bubble_file = os.getcwd()+'/saved_bubbles/v_n0_3_a_n1_1.p'
    #make_bubble_figure(bubble_file)
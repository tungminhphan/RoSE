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
def traces_to_animation(the_map, filename):
    # extract out traces from pickle file
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)

    global ax
    fig, ax = plt.subplots()

    # plot out agents and traffic lights
    for t in range(max(traces.keys())+1): 
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
def plot_map(map):
    x_min, x_max, y_min, y_max = get_map_corners(map)
    ax.axis('equal')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.grid()

    # fill in the obstacle regions
    for obs in map.non_drivable_nodes:
        rect = patches.Rectangle((obs[1],obs[0]), 1,1,linewidth=1,facecolor='k')
        ax.add_patch(rect)
    
    plt.gca().invert_yaxis()
    plt.axis('off')

def draw_car(agent_state_tuple):
    # global params
    x, y, theta, color, bubble = agent_state_tuple
    theta_d = Car.convert_orientation(theta)
    car_fig = Image.open(car_figs[color])
    car_fig = car_fig.rotate(theta_d, expand = False)
    ax.imshow(car_fig, zorder=1, interpolation='none', extent=[y, y+1, x, x+1])

    for grid in bubble:
        rect = patches.Rectangle((grid[1],grid[0]),1,1,linewidth=0.5,facecolor='grey', alpha=0.1)
        ax.add_patch(rect)


# to plot a single bubble for the paper figure
def plot_bubble(bubble):
    global ax, fig
    fig, ax = plt.subplots()
    ax.axis('equal')

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
    the_map = Map('./maps/straight_road', default_spawn_probability=0.1)
    #the_map = Map('map5', default_spawn_probability=0.05)
    output_filename = os.getcwd()+'/saved_traces/game.p'
    traces_to_animation(the_map, output_filename)

    #filename = os.getcwd()+'/saved_bubbles/v_n0_2_a_n2_2.p'
    #with open(filename, 'rb') as pckl_file:
    #    data = pickle.load(pckl_file)
    #plot_bubble(data[0])

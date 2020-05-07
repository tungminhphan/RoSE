'''
    File name: local_contracts.py
    Author: 潘明松
    Date created: 4/30/2020
'''
import os
import numpy as np
from collections import OrderedDict as od
import rose as rs
from ipdb import set_trace as st
import numpy as np
import math
import json as js
import _pickle as pickle

def translate_tile_list_by_xy(tile_list, xy):
    return [tuple(np.array(tile) + np.array(xy)) for tile in tile_list]

def find_directed_angle_between_vectors(vec1, vec2):
    dx = vec2[0]-vec1[0]
    dy = vec2[1]-vec1[1]
    d1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    d2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    cos_val = (vec1[0]*vec2[0] + vec1[1]*vec2[1])/(d1*d2)
    sin_val = (vec1[0]*vec2[1] - vec1[1]*vec2[0])/(d1*d2)
    return np.arctan2(sin_val, cos_val)

class LocalGridderFrame:
    def __init__(self, radius):
        self.radius = radius
        self.rel_tiles = self.get_rel_tiles()

    def get_rel_tiles(self):
        N = self.radius
        # using set to get rid of repetitions
        reference_tile_set = set()
        for i in range(-N, N+1):
            reference_tile_set.add((i,0))
        for j in range(-N, N+1):
            reference_tile_set.add((0,j))
        reference_tile_list = list(reference_tile_set)
        reference_tile_list.sort()
        return tuple(reference_tile_list)

    def get_goal_compass_dir(self, agent):
        x, y = agent.state.x, agent.state.y
        goal, _ = agent.supervisor.get_next_goal_and_plan()
        x_goal, y_goal = goal
        dx = np.sign(x_goal-x)
        dy = np.sign(y_goal-y)
        return (dx, dy)

    def get_mask(self, agent):
        rel_tiles = self.rel_tiles
        compass_dir = self.get_goal_compass_dir(agent)
        if sum(np.abs(compass_dir)) == 2: # if diagonal
            angle_of_rotation = find_directed_angle_between_vectors(compass_dir, [1, 1])
        elif sum(np.abs(compass_dir)) == 1: # straight ahead
            angle_of_rotation = find_directed_angle_between_vectors(compass_dir, [1, 0])

        rotated_tile_list = []
        for tile in self.rel_tiles:
            # rotate to match
            new_tile = tuple(rs.rotate_vector(tile, angle_of_rotation))
            rotated_tile_list.append(new_tile)
        mask = translate_tile_list_by_xy(rotated_tile_list, (agent.state.x, agent.state.y))
        return tuple(mask)

    def get_signature(self, game, agent):
        mask = self.get_mask(agent)
        signature = []
        for tile in mask:
            # if tile is in map at all
            if tile == (agent.state.x, agent.state.y): # if at ego
                feature = 'self'
            elif tile in agent.supervisor.game.map.nodes:
                if tile in agent.supervisor.game.occupancy_dict:
                    feature = 'occupied'
                else:
                    feature = 'free'
            else: # out of bounds
                feature = 'out'
            signature.append(feature)
        return tuple(signature), mask

class LocalContract:
    def __init__(self, name, frame):
        self.name = name
        self.frame = frame
        if not os.path.exists(self.name):
            self.contract_draft = dict()
        else: # load dictionary if already exists
            self.contract_draft = self.load_contract()

    def forbid_action(self, action):
        if signature not in self.contract_draft:
            self.contract_draft[signature] = [action]
        else:
            self.contract_draft[signature].append(action)

    def load_contract(self):
        with open(self.name, 'rb') as f:
            contract_draft = pickle.load(f)
        return contract_draft

    def dump_contract(self):
        with open(filename, 'wb') as f:
            pickle.dump(self.contract_draft, f)

if __name__ == '__main__':
    # specify the map name
    map_name = '10x10_field'
    the_field = rs.Field('./maps/'+map_name)

    # intialize gridder objects
    gridder1 = rs.Gridder(x=0,y=0)
    gridder2 = rs.Gridder(x=9,y=9)

    # specify gridder objects' goals
    goal1 = (gridder2.state.x, gridder2.state.y)
    goal2 = (gridder1.state.x, gridder1.state.y)

    # create agent set
    agent_set = [gridder1, gridder2]

    # create game from map and agent set
    the_game = rs.Simulation(the_map=the_field,agent_set=agent_set)

    # define local frames
    gridder_frame = LocalGridderFrame(radius=3)

    # define the local contract object
    contract = LocalContract(name='test', frame=gridder_frame)

    # define and set supervisors for the robots
    contract_supervisor1 = rs.LocalContractSupervisor(the_game, goal1, contract)
    gridder1.set_supervisor(contract_supervisor1)
    contract_supervisor2 = rs.LocalContractSupervisor(the_game, goal2, contract)
    gridder2.set_supervisor(contract_supervisor2)

    compass_controller = rs.CompassController()
    gridder1.set_controller(compass_controller)
    gridder2.set_controller(compass_controller)

    game = rs.ContractGame(the_map=the_field, agent_set=agent_set)
    game.learn()

#    sim.play()

#    vec0 = [1, -1]
#    vec1 = [-1, 1]
#    rot = find_directed_angle_between_vectors(vec0, vec1)
#    print(rot/np.pi * 180)
#
#    print(vec0)
#    print(vec1)
#    print('after')
#    vec0 = rs.rotate_vector(vec0, rot)
#    print(vec0)

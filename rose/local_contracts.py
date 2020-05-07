'''
    File name: local_contracts.py
    Authors: Tung Phan
    Date created: 4/30/2020
'''
import numpy as np
from collections import OrderedDict as od
import rose as rs
from ipdb import set_trace as st
import numpy as np
import math

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

class LocalFrame:
    def __init__(self, agent, radius):
        self.agent = agent
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

    def get_goal_compass_dir(self):
        x, y = self.agent.state.x, self.agent.state.y
        goal, _ = self.agent.supervisor.get_next_goal_and_plan()
        x_goal, y_goal = goal
        dx = np.sign(x_goal-x)
        dy = np.sign(y_goal-y)
        return (dx, dy)

    def get_mask(self):
        rel_tiles = self.rel_tiles
        compass_dir = self.get_goal_compass_dir()
        if sum(np.abs(compass_dir)) == 2: # if diagonal
            angle_of_rotation = find_directed_angle_between_vectors(compass_dir, [1, 1])
        elif sum(np.abs(compass_dir)) == 1: # straight ahead
            angle_of_rotation = find_directed_angle_between_vectors(compass_dir, [1, 0])

        rotated_tile_list = []
        for tile in self.rel_tiles:
            # rotate to match
            new_tile = tuple(rs.rotate_vector(tile, angle_of_rotation))
            rotated_tile_list.append(new_tile)
        return tuple(rotated_tile_list)

    def get_signature(self, game):
        mask = self.get_mask()
        signature = []
        for tile in mask:
            for tile in agent.supervisor.game.map:
                feature = None
                signature.append(feature)
        return signature

class LocalContractSupervisor:
    def __init__(self, local_frame):
        self.frame_to_forbidden_actions = od()
        self.local_frame = local_frame

    def forbid_action(self, action):
        if frame not in self.frame_to_forbidden_actions:
            self.frame_to_forbidden_actions[frame] = [action]
        else:
            self.frame_to_forbidden_actions[frame].append(action)

if __name__ == '__main__':
    map_name = '10x10_field'
    the_field = rs.Field('./maps/'+map_name)
    gridder1 = rs.Gridder(x=0,y=0)
    gridder2 = rs.Gridder(x=9,y=9)
    goal1 = (gridder2.state.x, gridder2.state.y)
    goal2 = (gridder1.state.x, gridder1.state.y)

    agent_set = [gridder1, gridder2]
    the_game = rs.Simulation(the_map=the_field,agent_set=agent_set)

    local_frame1 = LocalFrame(agent=gridder1, radius=3)
    local_frame2 = LocalFrame(agent=gridder2, radius=3)

    contract_supervisor1 = rs.LocalContractSupervisor(the_game, goal1, local_frame1)
    gridder1.set_supervisor(contract_supervisor1)
    contract_supervisor2 = rs.LocalContractSupervisor(the_game, goal2, local_frame2)
    gridder2.set_supervisor(contract_supervisor2)

    compass_controller = rs.CompassController()
    gridder1.set_controller(compass_controller)
    gridder2.set_controller(compass_controller)

    sim = rs.Simulation(the_map=the_field, agent_set=agent_set)
    sim.animate(0.1)
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

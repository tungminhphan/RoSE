from collections import OrderedDict as od
import rose as rs
from ipdb import set_trace as st
import numpy as np

map_name = '10x10_field'
the_field = rs.Field('./maps/'+map_name)
gridder1 = rs.Gridder(x=0,y=0)
goal1 = (9, 9)
gridder2 = rs.Gridder(x=9,y=9)
goal2 = (0, 0)

contract_controller1 = rs.LocalContractController(the_game, goal1)
contract_controller2 = rs.LocalContractController(the_game, goal1)

class LocalFrame:
    def __init__(self, agent, radius):
        self.agent = agent
        self.radius = radius

    def get_tiles(self, agent, N):
        x, y = self.agent.state.x, self.agent.state.y
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                print(i,j)

    def __equal__(self):
        pass

class LocalContract:
    def __init__(self, local_frame):
        self.frame_to_forbidden_actions = od()
        self.local_frame = local_frame

    def forbid_action(self, action):
        if frame not in self.frame_to_forbidden_actions:
            self.frame_to_forbidden_actions[frame] = [action]
        else:
            self.frame_to_forbidden_actions[frame].append(action)

def translate_tile_list_by_xy(tile_list, xy):
    return [tuple(np.array(tile) + np.array(xy)) for tile in tile_list]

#x, y = agent.state.x, agent.state.y


def get_frame(agent):
    x, y = agent.state.x, agent.state.y
    x_goal, y_goal = agent.supervisor.goals
    dx = np.sign(x_goal-x)
    dy = np.sign(y_goal-y)



#def get_rel_tiles(agent, N):
#    reference_tile_set = set()
#    for i in range(-N, N+1):
#        reference_tile_set.add((i,0))
#    for j in range(-N, N+1):
#        reference_tile_set.add((0,j))
#    reference_tile_list = list(reference_tile_set)
#    reference_tile_list.sort()
#    tile_sets = [translate_tile_list_by_xy(reference_tile_list, (x,y))]
#    angles = [np.pi/2 * k for k in range(1,4)]
#    for angle in angles:
#        rotated_tile_list = []
#        for tile in reference_tile_set:
#            new_tile = tuple(rs.rotate_vector(tile, np.pi))
#            rotated_tile_list.append(new_tile)
#        tile_sets.append(translate_tile_list_by_xy(rotated_tile_list,(x,y)))
#    st()
#    return tile_sets


#print(get_rel_tiles(gridder1, 2))
#print(len(get_rel_tiles(gridder1, 2)))

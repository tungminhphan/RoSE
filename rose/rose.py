'''

    File name: rose.py
    Authors: Tung Phan, Karena Cai
    Date created: 1/10/2020
'''
from typing import List, Any
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import curses
import time
import sys
import random
import math
import numpy as np
from ipdb import set_trace as st
import csv
import networkx as nx
from collections import OrderedDict as od
from collections import namedtuple
import os
import _pickle as pickle
import copy as cp
#from plotting import traces_to_animation

car_colors = ['blue', 'brown', 'gray', 'green', 'light_green', 'orange', 'red', 'yellow']
IOMap = namedtuple('IOMap', ['sources','sinks','map'])
TLNode = namedtuple('TLNode', ['xy', 'green', 'yellow', 'red'])
Intersection = namedtuple('Intersection', ['tiles', 'pcorner', 'mcorner', 'height', 'width'])
Neighbor = namedtuple('Neighbor', ['xyt', 'weight', 'name'])
DIRECTION_TO_VECTOR = {'east': [0,1], 'west': [0,-1], 'north': [-1,0], 'south': [1,0]}

def rotate_vector(vec, theta):
    """
    take in a 2D vector and an angle in radians and outputs the same vector
    rotated by the amount specified by the input angle in a CCW fashion.
    """
    # create rotation matrix
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.array([int(round(x)) for x in np.matmul(rot_mat, vec)])

class Drawable:
    """
    A class for terminal drawable objects

    Attributes:
        xy : tuple[int]
            the x, y coordinates of the object
        drawstr: str
            the string representation of the object
    """
    def __init__(self, xy, drawstr):
        self.xy = xy
        self.drawstr = drawstr

class AgentState:
    def __init__(self, agent_name, state_variable_names, **kwargs):
        self.state_variable_names = state_variable_names
        self.set_state_variables(kwargs)
        self.agent_name = agent_name

    def set_state_variables(self, kwargs):
        for state_var in self.state_variable_names:
            exec('self.%s = kwargs.get(\'%s\')' % (state_var, state_var))

    def __str__(self):
        printed = []
        for state_var in self.state_variable_names:
            printed.append(state_var + ' = ' + str(eval('self.%s' % state_var)))
        return self.agent_name + 'State(' + ', '.join(printed) + ')'

    def __tuple__(self):
        tup = ()
        for i, state_var in enumerate(self.state_variable_names):
            a = eval('self.%s' % (state_var))
            tup = (*tup, a)
        return tup

    def __eq__(self, other):
        return self.__tuple__()  == other.__tuple__()

class Agent:
    def set_controller(self, controller):
        self.controller = controller

    def set_supervisor(self, supervisor):
        supervisor.set_plant(self)

    def get_id(self):
        return id(self)

    @classmethod
    def hack_state(cls, state, **kwargs):
        new_state = AgentState(agent_name=state.agent_name, state_variable_names=state.state_variable_names)
        for name in new_state.state_variable_names:
            # just copy from state
            is_not_none = eval('kwargs.get(\'%s\') != None' % name)
            exec('new_state.%s = state.%s' % (name, name))
            if is_not_none:
                # update from is_not_none input
                exec('new_state.%s = kwargs.get(\'%s\')' % (name, name))
        return new_state

    def __init__(self, **kwargs):
        if 'agent_name' in kwargs:
            self.agent_name = kwargs.get('agent_name')
        else:
            self.agent_name = 'Agent'
        if 'agent_color' in kwargs:
            self.agent_color = kwargs.get('agent_color')
        else:
            self.agent_color = self.get_random_color()
        self.attributes = kwargs.get('attributes')
        self.state_variable_names = kwargs.get('state_variable_names')
        self.set_state(kwargs)
        self.set_attributes(kwargs)
        if 'controller' in kwargs:
            self.controller = kwargs.get('controller')
        else:
            self.controller = None
        if 'supervisor' in kwargs:
            self.supervisor = kwargs.get('supervisor')
        else:
            self.supervisor = None

    # return a random color from an array
    def get_random_color(self):
        ind = random.randrange(len(car_colors))
        return car_colors[ind]

    def get_symbol(self):
        if not self.symbol:
            return 'o'
        else:
            return self.symbol

    def run(self):
        assert self.controller, "Agent has no controller"
        #self.set_intention()
        self.controller.run_on(self)
        if self.supervisor:
            self.supervisor.run()

    def set_attributes(self, kwargs_dict):
        for attr in self.attributes:
            exec('self.%s = kwargs_dict.get(\'%s\')' % (attr, attr))

    def set_state(self, kwargs_dict):
        if 'state' in kwargs_dict:
            new_state = kwargs_dict.get('state')
        else:
            new_state = AgentState(**kwargs_dict)
        self.state = new_state

    def query_occupancy(self, ctrl, state=None):
        raise NotImplementedError

    def query_next_state(self, ctrl):
        assert ctrl in self.get_all_ctrl()
        return self.query_occupancy(ctrl)[-1]

    def apply(self, ctrl):
        prior_state = self.state
        # check for collision with any of the other agents
        self.check_collision(ctrl)
        self.state = self.query_next_state(ctrl)
        self.supervisor.game.update_occupancy_dict()
        # save action that was chosen
        self.ctrl_chosen = ctrl
        # save the agents currently in agent bubble
        self.agents_in_bubble = self.find_agents_in_bubble()
        # check wehther the state is out of bounds
        self.check_out_of_bounds(self, prior_state, ctrl, self.state)
        # check whether the updated joint state is safe
        chk_joint_safety = self.check_joint_state_safety(return_list=True)
        self.supervisor.game.unsafe_joint_state_dict[self.supervisor.game.time] = self.check_joint_state_safety(return_list=True)
        


class Gridder(Agent):
    def __init__(self, **kwargs):
        agent_name = 'Gridder'
        state_variable_names = ['x', 'y']
        attributes = []
        super(Gridder, self).__init__(attributes=attributes, agent_name=agent_name, state_variable_names=state_variable_names, **kwargs)

    def get_symbol(self):
        return '🤖'

    def get_all_ctrl(self, state=None):
        return ['up', 'down', 'left', 'right', 'stay']

    def query_occupancy(self, ctrl, state=None):
        if state == None:
            state = self.state
        if ctrl == 'up':
            next_state = Agent.hack_state(state, y = state.y + 1)
        elif ctrl == 'down':
            next_state = Agent.hack_state(state, y = state.y - 1)
        elif ctrl == 'left':
            next_state = Agent.hack_state(state, x = state.x - 1)
        elif ctrl == 'right':
            next_state = Agent.hack_state(state, x = state.x + 1)
        else:
            next_state = Agent.hack_state(state)
        occupancy_list = [next_state]
        return occupancy_list

class Car(Agent):
    def __init__(self, **kwargs):
        agent_name = 'Car'
        attributes = ['v_min', 'v_max', 'a_min', 'a_max']
        state_variable_names = ['x', 'y', 'heading', 'v']
        super(Car, self).__init__(attributes=attributes, agent_name=agent_name, state_variable_names=state_variable_names, **kwargs)
        self.acc_vals = np.arange(self.a_min, self.a_max+1)
        self.default_state = Car.hack_state(self.state, x=0, y=0, heading='east', v=0)
        self.default_bubbles = self.get_default_bubbles()
        self.send_conflict_requests_to = [] # list of agents
        self.received_conflict_requests_from = [] # list of agents
        self.agent_max_braking_not_enough = None
        self.token_count = 0
        self.intention = None

        # attributes for saving agent info
        self.spec_struct_trace = od()
        self.ctrl_chosen = None
        self.unsafe_joint_state_dict = {}
        self.agents_in_bubble = []
        self.straight_action_eval = {}
        self.action_selection_flags = ()
        #self.lead_vehicle = None
        #self.lead_agent = None

    def get_intention(self):
        return self.intention

    def set_debug_intention(self, intention):
        self.intention = intention

    # only works when used with specification structure controller
    def set_intention(self):
        def ctrl_dict_to_tuple(ctrl):
            return (ctrl['steer'], ctrl['acceleration'])
        scores = []
        all_ctrls = self.get_all_ctrl()
        spec_struct_trace = od()

        for ctrl in all_ctrls:
            score = 0
            scores_sv = od()
            for oracle in self.controller.specification_structure.oracle_set:
                o_score = oracle.evaluate(ctrl, self, self.supervisor.game)
                o_tier = self.controller.specification_structure.tier[oracle]
                if oracle.name != 'backup_plan_safety':
                    try:
                       score += int(o_score) * self.controller.specification_structure.tier_weights[o_tier]
                    except:
                        pass
                    scores_sv[oracle.name] = o_score

            scores.append(score)

            scores_sv['total'] = score
            spec_struct_trace[ctrl_dict_to_tuple(ctrl)] = scores_sv

        choice = random.choice(np.where(scores == np.max(scores))[0])
        self.intention = all_ctrls[choice]
        self.spec_struct_trace = spec_struct_trace

    def set_token_count(self, cnt):
        self.token_count = cnt

    # reset conflict send and receive lists
    def reset_conflict_lists(self):
        self.agent_max_braking_not_enough = None
        self.send_conflict_requests_to = []
        self.received_conflict_requests_from = []

    # conflict resolution, returns True if winner and false if not winner
    def check_conflict_resolution_winner(self, specify_agent=False):
        # collect all agents in send and receive requests and find the winner
        conflict_cluster = list(set(self.send_conflict_requests_to + self.received_conflict_requests_from + [self]))

        max_val = 0
        max_agent_list = []
        for agent in conflict_cluster:
            if agent.token_count > max_val:
                max_agent_list = [agent]
            elif agent.token_count == max_val:
                max_agent_list.append(agent)
        # resolve ties with max values with agent ID comparison
        ind_max = np.argmax(np.array([agent.get_id() for agent in max_agent_list]))
        agent_winner = max_agent_list[ind_max]
        if specify_agent:
            return agent_winner.get_id() == self.get_id(), agent_winner
        else:
            return agent_winner.get_id() == self.get_id()

    def get_length_along_bundle(self):
        assert self.supervisor, 'Supervisory controller required!'
        x, y = self.state.x, self.state.y
        current_bundles = self.supervisor.game.map.tile_to_bundle_map[x,y]
        for bundle in current_bundles:
            if bundle.direction == self.state.heading:
                return bundle.tile_to_relative_length((x,y)), bundle

    def get_width_along_bundle(self):
        x, y = self.state.x, self.state.y
        current_bundles = self.supervisor.game.map.tile_to_bundle_map[x,y]
        for bundle in current_bundles:
            if bundle.direction == self.state.heading:
                return bundle.tile_to_relative_width((x,y)), bundle

    def get_default_bubbles(self):
        # caching the agent bubbles
        def create_pickle_file(filename):
            vel = np.arange(self.v_min, self.v_max+1)
            bubble_dict = dict()
            for v in vel:
                bubble_dict[v] = self.get_default_bubble(v)
            with open(filename, 'wb+') as pckl_file:
                pickle.dump(bubble_dict, pckl_file)
            pass

        # string assumes v_min and a_min are negative
        car_param = 'v_' + 'n' + str(abs(self.v_min)) + '_' + str(self.v_max) + \
            '_a_n' + str(abs(self.a_min)) + '_' + str(self.a_max)+'.p'
        # check if file exists
        output_dir = os.getcwd()+'/saved_bubbles/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filedir = output_dir + car_param #v_n1_2_a_n2_2.p'
        if not os.path.exists(filedir):
            create_pickle_file(filedir)

        with open(filedir, 'rb') as pckl_file:
            data = pickle.load(pckl_file)
        return data

    # assuming backup plan is to brake maximally
    def get_maximum_braking_controls(self):
        def at_complete_stop(state):
            return state.v == 0
        plan = dict()
        plan['controls'] = {'acceleration': self.a_min, 'steer': 'straight'}
        plan['v_min'] = 0
        plan['v_max'] = self.v_max
        plan['stopping_condition'] = at_complete_stop
        return plan

    def query_backup_plan(self, backup_plan=None, state=None):
        if backup_plan == None:
            backup_plan = self.get_maximum_braking_controls()
        if not state:
            state = self.state
        tile_sequence_chain = [(0, [(state.x, state.y)])]
        step = 1
        while not backup_plan['stopping_condition'](state):
            occupancy_list = Car.query_class_occupancy(backup_plan['controls'], state, backup_plan['v_min'], backup_plan['v_max'], inverse=False)
            required_tiles = [(state.x, state.y) for state in occupancy_list]
            tile_sequence_chain.append((step, required_tiles))
            state = occupancy_list[-1]
            step += 1
        return tile_sequence_chain

    def get_drawables(self):
        drawables = []
        drawable = Drawable([self.state.x, self.state.y], self.get_symbol())
        drawables.append(drawable)
        return drawables

    def get_symbol(self):
        if self.state.heading == 'west':
            return '🠴'
        elif self.state.heading == 'east':
            return '🠶'
        elif self.state.heading == 'north':
            return '🠵'
        elif self.state.heading == 'south':
            return '🠷'

    @classmethod
    def convert_orientation(cls, inp):
        if inp == 'east':
            return 0
        elif inp == 'north':
            return 90
        elif inp == 'west':
            return 180
        elif inp == 'south':
            return 270
        elif inp%360 == 0:
            return 'east'
        elif inp%360 == 90:
            return 'north'
        elif inp%360 == 180:
            return 'west'
        elif inp%360 == 270:
            return 'south'
        else:
            raise TypeError('Check your input!')

    @classmethod
    def get_all_class_ctrl(cls, state, acc_vals, inverse=False, straight_only=False):
        all_ctrl = []
        def make_ctrl(acc, steer):
            return {'steer': steer, 'acceleration': acc}
        
        for acc in acc_vals:
            if state.v + acc == 1 and not straight_only:
                steer_vals = ['straight', 'left-lane', 'right-lane', 'left-turn', 'right-turn']
            else:
                steer_vals  = ['straight']
            for steer in steer_vals:
                all_ctrl.append(make_ctrl(acc, steer))
        return all_ctrl

    @classmethod
    def query_class_occupancy(cls, ctrl, state, v_min, v_max, inverse=False):
        acc = ctrl['acceleration']
        steer = ctrl['steer']

        def get_new_heading(act):
            heading = Car.convert_orientation(state.heading)
            if act == 'left-turn':
                heading += 90
            elif act == 'right-turn':
                heading -= 90
            return Car.convert_orientation(heading)

        def relativize_absolute_displacement(abs_dis, inverse=False):
            rel_dis = []
            offset = np.array([state.x, state.y])
            for displacement in abs_dis:
                heading = Car.convert_orientation(state.heading) * np.pi/180
                rel_dis.append(offset + rotate_vector(displacement, heading))
                offset_reverse = rotate_vector(displacement, heading)
            if inverse:
                rel_dis = [x-offset_reverse for x in rel_dis]
            return rel_dis

        abs_displacements = []
        final_heading = state.heading
        init_heading = final_heading
        final_v = min(max(acc + state.v, v_min), v_max)
        if steer == 'straight':
            num_moves = final_v
            num_moves_sign = np.sign(num_moves)
            num_moves_abs = np.abs(num_moves)
            if num_moves_sign == 0:
                num_moves_sign = 1
            for k in np.arange(0, num_moves_sign * (num_moves_abs + 1), num_moves_sign):
                dx = 0
                dy = k
                abs_displacements.append(np.array([dx, dy]))
        elif steer == 'left-lane' and final_v == 1:
            abs_displacements = [[0, 0], [-1, 0], [-1, 1]]
        elif steer == 'left-turn' and final_v == 1:
            abs_displacements = [[0, 0], [0, 1], [-1, 1]]
            final_heading = get_new_heading('left-turn')
        elif steer == 'right-lane' and final_v == 1:
            abs_displacements = [[0, 0], [1, 0], [1, 1]]
        elif steer == 'right-turn' and final_v == 1:
            abs_displacements = [[0, 0], [0, 1], [1, 1]]
            final_heading = get_new_heading('right-turn')
        else:
            return None

        # assign occupancy list
        occupancy_list = [None]*len(abs_displacements)
        for i, xy in enumerate(relativize_absolute_displacement(abs_displacements, inverse=inverse)):
            heading = init_heading if i == 0 else final_heading
            vel = final_v if not inverse else state.v-acc
            if vel < v_min or vel > v_max: return None
            occupancy_list[i] = Car.hack_state(state, x=xy[0], y=xy[1], heading=heading, v=vel)

        return occupancy_list

    def query_occupancy(self, ctrl, state=None, inverse=False):
        if state is None: state = self.state
        return Car.query_class_occupancy(ctrl, state, self.v_min, self.v_max, inverse=inverse)

    def get_all_ctrl(self, state=None,inverse=False, straight_only=False):
        if state is None: state = self.state
        return Car.get_all_class_ctrl(state, self.acc_vals, inverse=inverse, straight_only=straight_only)

    def get_agents_with_higher_precedence_in_bubble(self):
        """
        gets the agents with higher precedence, assuming ego and all agents in
        agent_set belong to the same bundle
        """
        def get_agents_with_higher_precedence(agent_set):
            higher_pred = []
            ego_tile = self.state.x, self.state.y
            ego_heading = self.state.heading
            # TODO: fix this bandage code
            if (ego_tile, ego_heading) in self.supervisor.game.map.special_goal_tiles:
                ego_heading = Car.convert_orientation(Car.convert_orientation(ego_heading) + 90)

            bundle = self.supervisor.game.map.get_bundle_from_directed_tile(ego_tile, ego_heading)
            ego_score = bundle.tile_to_relative_length(ego_tile)
            for agent in agent_set:
                agent_tile = agent.state.x, agent.state.y
                try:
                    agent_score = bundle.tile_to_relative_length(agent_tile)
                except:
                    print("agent out of bounds, setting to have higher prec?")
                    print(agent_tile)
                    agent_score = np.inf
                if agent_score > ego_score:
                    higher_pred.append(agent)
            return higher_pred
        agents_in_bubble = self.find_agents_in_bubble()
        return get_agents_with_higher_precedence(agents_in_bubble)

    # check farthest straight agent can go forward (assuming agent in front already took its turn)
    def get_max_forward_ctrl(self):
        lead_agent = self.find_lead_agent()
        if lead_agent is None:
            ctrl = {'acceleration': self.a_max, 'steer': 'straight'}
            return ctrl
        x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
        # try all acceleration values
        ctrl_acc = np.arange(self.a_max, self.a_min-1, -1)
        for acc_val in ctrl_acc:
            ctrl = {'acceleration': acc_val, 'steer':'straight'}
            occ = self.query_occupancy(ctrl)
            intersection = self.check_occupancy_intersection(occ, [lead_agent.state])
            safe_chk = self.check_safe_config(self, lead_agent, st_1=occ[-1], st_2=lead_agent.state)
            if safe_chk and not intersection: return ctrl
        return self.get_backup_plan_ctrl()
    
    # get the best straight action agent can do
    def get_best_straight_action(self):
        def ctrl_dict_to_tuple(ctrl):
            return (ctrl['steer'], ctrl['acceleration'])
        # evaluate all the straight actions with the oracle
        straight_action_eval = {}
        scores = []
        all_straight_ctrl = self.get_all_ctrl(straight_only=True)
        for ctrl in all_straight_ctrl:
            score_save = {}
            score = 0
            for oracle in self.controller.specification_structure.oracle_set:
                o_score = oracle.evaluate(ctrl, self, self.supervisor.game)
                o_tier = self.controller.specification_structure.tier[oracle]
                try:
                    score += int(o_score) * self.controller.specification_structure.tier_weights[o_tier]
                except:
                    pass
                score_save[oracle.name] = o_score
            scores.append(score)
            score_save['total'] = score
            straight_action_eval[ctrl_dict_to_tuple(ctrl)] = score_save

        choice = random.choice(np.where(scores == np.max(scores))[0])
        self.straight_action_eval = straight_action_eval
        return all_straight_ctrl[choice]

    #=== method for action selection strategy==========================#
    # defines which action to select given precedence list and whether or not
    # agent won in its conflict cluster
    def action_selection_strategy(self):
        # figure out whether agent is sender, receiver, both or neither
        def get_agent_type():
            len_from = len(self.received_conflict_requests_from)
            len_to = len(self.send_conflict_requests_to)
            if len_from > 0 and len_to == 0:
                return "receiver"
            elif len_from == 0 and len_to > 0:
                return "sender"
            elif len_from > 0 and len_to > 0:
                return "both"
            else:
                return "none"

        # figure out whether intended action conflicts with agents that have
        # higher precedence
        def check_conflict_with_higher_precedence_agents():
            my_occ = self.query_occupancy(self.intention)
            # presumably agents with higher precedence are only ones in agent bubble
            for agent in self.get_agents_with_higher_precedence_in_bubble():
                # check whether intention overlaps with agents of higher precedence
                # and intended action has final config that preserves back-up plan
                grid_pt = [(agent.state.x, agent.state.y)]
                occupancy_overlap = self.check_occupancy_intersection(my_occ, grid_pt)
                safe_config = self.check_safe_config(self, agent, my_occ[-1], agent.state)
                if occupancy_overlap or not safe_config:
                    return True
            return False

        # TODO: need to change assuming other agent has already gone
        # check max amount another agent needs to yield for another agent to change lanes
        def check_min_dec_yield_req(winning_agent):
            return self.get_backup_plan_ctrl()

        # check whether received request from winning agent
        def chk_receive_request_from_winning_agent(winning_agent):
            for agent in self.received_conflict_requests_from:
                if winning_agent.get_id() == agent.get_id():
                    return True
            return False

        # switch statements that define the outcome of the decision tree
        agent_type = get_agent_type()
        # True means no conflict with agents in bubble with higher precedence
        bubble_chk = not check_conflict_with_higher_precedence_agents()
        # True means winner of conflict cluster
        cluster_chk, winning_agent = self.check_conflict_resolution_winner(specify_agent=True)
        chk_receive_request_from_winner = chk_receive_request_from_winning_agent(winning_agent)
        max_braking_enough = self.agent_max_braking_not_enough is None

        self.action_selection_flags = (agent_type, bubble_chk, cluster_chk, max_braking_enough)
        #self.lead_vehicle = self.find_lead_agent()
        #if self.find_lead_agent():
            #self.lead_vehicle = self.find_lead_agent().state.__tuple__()

        # list of all possible scenarios and what action to take
        if agent_type == 'none' and bubble_chk and max_braking_enough:
            # take intended action (all checks pass)
            ctrl = self.intention
            self.token_count = 0
        elif agent_type == 'none' and not bubble_chk and max_braking_enough:
            # TODO: take straight action, best safe one that aligns with intention
            self.token_count = self.token_count+1
            ctrl = self.get_best_straight_action()
        elif agent_type == 'none' and not max_braking_enough:
            self.token_count = self.token_count+1
            ctrl = self.get_best_straight_action()

        elif agent_type == 'sender' and not cluster_chk:
            # TODO: take straight action, best safe one that aligns with intention
            self.token_count = self.token_count+1
            ctrl = self.get_best_straight_action()
        elif agent_type == 'sender' and cluster_chk and not max_braking_enough:
            # max braking not enough
            ctrl = self.get_best_straight_action()
            self.token_count = self.token_count+1
        elif agent_type == 'sender' and cluster_chk and max_braking_enough and not bubble_chk:
            # TODO: take straight action, best safe one that aligns with intention
            self.token_count = self.token_count+1
            ctrl = self.get_best_straight_action()
        elif agent_type == 'sender' and cluster_chk and max_braking_enough and bubble_chk:
            # take intended action!!
            self.token_count = 0
            ctrl = self.intention
        
        elif agent_type == 'both' and not cluster_chk:
            # TODO: take straight action, best safe one that aligns with intention
            self.token_count = self.token_count+1
            ctrl = check_min_dec_yield_req(winning_agent)
        elif agent_type == 'both' and cluster_chk and not max_braking_enough:
            # max braking not enough
            ctrl = self.get_best_straight_action()
            self.token_count = self.token_count+1
        elif agent_type == 'both' and cluster_chk and max_braking_enough and not bubble_chk:
            # TODO: take straight action, best safe one that aligns with intention
            self.token_count = self.token_count+1
            ctrl = self.get_best_straight_action()
        elif agent_type == 'both' and cluster_chk and max_braking_enough and bubble_chk:
            # take intended action!!
            self.token_count = 0
            ctrl = self.intention
        
        elif agent_type == 'receiver' or agent_type and bubble_chk and not cluster_chk:
            # yield as much as needed for conflict winner to move
            # assumes winner has already taken its action!!!
            ctrl = check_min_dec_yield_req(winning_agent)
            self.token_count = self.token_count+1
        elif agent_type == 'receiver' and bubble_chk and cluster_chk:
            # if agent_type is receiver, then take intended action as long as safe w.r.t agents behind in precedence too
            ctrl = self.intention
            self.token_count = 0
        elif agent_type == 'receiver' and not bubble_chk and cluster_chk:
            # TODO: take straight action, best safe one that aligns with intention
            ctrl = self.get_best_straight_action()
            self.token_count = self.token_count+1
        

        else:
            print(agent_type)
            print(bubble_chk)
            print(cluster_chk)
            print("Error: invalid combination of inputs to action selection strategy!")
            ctrl = None

        return ctrl

    #=== methods for car bubble =======================================#
    # get agent bubble (list of grid points)
    def get_bubble(self, the_map=None):
        default_bubble = self.default_bubbles[self.state.v]
        dx, dy, dtheta = self.find_transformation(self.default_state.__tuple__(), self.state.__tuple__())
        # transform to current car state
        bubble_tf = [self.transform_state(node, dx, dy, dtheta, assign_heading=False) for node in default_bubble]
        # only keep nodes that are in the drivable set
        if the_map: bubble_tf = [node for node in bubble_tf if node in the_map.drivable_nodes]
        return list(set(bubble_tf))

    # find all agents in the agents' bubble
    def find_agents_in_bubble(self):
        occupancy_dict = self.supervisor.game.occupancy_dict
        agents_in_bubble = [occupancy_dict[grid_pt] for grid_pt in \
            self.get_bubble() if grid_pt in occupancy_dict]
        return agents_in_bubble

    #=== find agents in this agents' conflict cluster ===================#
    def find_agents_to_send_conflict_request(self):
        #__import__('ipdb').set_trace(context=21)
        send_requests_list = []
        # collect all agents in bubble
        agents_in_bubble = self.find_agents_in_bubble()
        # if agent intention is to go straight, it shouldn't send a request
        if self.intention['steer'] == 'straight': return send_requests_list

        # check whether agent is in conflict with other agents in its bubble
        #print("AGENT CHECKING CONFLICT")
        #print(self.state)
        for agent in agents_in_bubble:
            if agent.get_id() != self.get_id():
                # first check if agent is longitudinally equal or ahead of other agent
                try:
                    chk_lon = (self.get_length_along_bundle()[0]-agent.get_length_along_bundle()[0])>=0
                except:
                    return []
                if chk_lon:
                    #print("chk_lon passed")
                    # send request to agent behind if intentions conflict
                    chk_to_send_request = self.check_to_send_conflict_request(agent)
                    if chk_to_send_request: send_requests_list.append(agent)
                    # check whether max yield is not enough; if not, set flag
                    chk_max_braking_not_enough = self.intention_bp_conflict(agent)
                    if chk_max_braking_not_enough:
                        #print("max braking not enough")
                        self.agent_max_braking_not_enough = agent
                        return []

        return send_requests_list

    #======add in code for checking whether agent can go for right turn=========#
    def check_right_turn_is_clear(self, right_turn_ctrl):
        # get the bundle that agent is trying to turn into
        #right_turn_ctrl = {'acceleration': 1-self.state.v, 'right-turn'}
        next_st = self.query_occupancy(right_turn_ctrl)[-1]
        bundle = self.supervisor.game.map.get_bundle_from_directed_tile((next_st.x, next_st.y), next_st.heading)
        if bundle is None: 
            st()
        # collect all agents in agent bundle AND in agent bubble
        for agent in self.find_agents_in_bubble():
            # get the bundle the agent in the bubble is in 
            agent_bundle = agent.supervisor.game.map.get_bundle_from_directed_tile((agent.state.x, agent.state.y), agent.state.heading)
            if agent_bundle is not None: 
                # if agent passes these checks, then see whether the agents are in conflict 
                if bundle.get_id() == agent_bundle.get_id() and agent.get_id() != self.get_id(): 
                    # check whether an agent takes max acc and self takes right turn are valid actions
                    max_acc_ctrl = {'acceleration': agent.a_max, 'steer': 'straight'}
                    chk_valid_1 = self.check_valid_actions(self, right_turn_ctrl, agent, max_acc_ctrl)
                    # check whether an agent takes back-up and self takes right turn are valid actions
                    chk_valid_2 = self.check_valid_actions(self, right_turn_ctrl, agent, agent.get_backup_plan_ctrl())
                    if not (chk_valid_1 and chk_valid_2): 
                        return False
            else:
                pass
        return True

    def check_collision_in_bubble(self, ctrl):
        # get agents in bubble
        agents_in_bubble = self.find_agents_in_bubble()
        all_agent_gridpts = [(agent.state.x, agent.state.y) for agent in agents_in_bubble if agent.get_id()!=self.get_id()]

        # 
        occ = self.query_occupancy(ctrl)
        action_gridpts = [(state.x, state.y) for state in self.query_occupancy(ctrl)]
        gridpts_intersect = list(set(all_agent_gridpts) & set(action_gridpts))
        return len(gridpts_intersect) > 0
    
    #============verifying agent back-up plan invariance===================#

    # check for collision with occupancy dict
    def check_collision(self, ctrl):
        # collect all grid points from occupancy dict except for own agent
        all_agent_gridpts = [gridpt for gridpt, agent in self.supervisor.game.occupancy_dict.items() if agent.get_id()!=self.get_id()]
        occ = self.query_occupancy(ctrl)
        if occ is None:
            return True
        else:
            action_gridpts = [(state.x, state.y) for state in self.query_occupancy(ctrl)]
        gridpts_intersect = list(set(all_agent_gridpts) & set(action_gridpts))
        collision_check = len(gridpts_intersect) > 0
        if collision_check:
            ag = self.supervisor.game.occupancy_dict[gridpts_intersect[0]]
            if self.supervisor.game.time not in self.supervisor.game.collision_dict:
                self.supervisor.game.collision_dict[self.supervisor.game.time] = \
                    [(self.get_id(), self.state.__tuple__(), self.intention, ag.get_id(), ag.state.__tuple__(), ag.intention)]
            else:
                self.supervisor.game.collision_dict[self.supervisor.game.time].append((self.get_id(),self.state.__tuple__(), \
                    self.intention, ag.get_id(), ag.state.__tuple__(), ag.intention))
            print(self.state.__tuple__(), self.intention, self.agent_color, ag.state.__tuple__(), ag.intention, ag.agent_color)
        return len(gridpts_intersect) > 0
    
    def check_out_of_bounds(self, agent, prior_state, ctrl, state):
        out_of_bounds_chk = (state.x, state.y) not in self.supervisor.game.map.drivable_nodes
        if out_of_bounds_chk:
            if self.supervisor.game.time not in self.supervisor.game.out_of_bounds_dict:
                self.supervisor.game.out_of_bounds_dict[self.supervisor.game.time] = \
                    [(self.get_id(), prior_state.__tuple__(), ctrl, self.state.__tuple__())]
            else:
                self.supervisor.game.out_of_bounds_dict[self.supervisor.game.time].append((self.get_id(), prior_state.__tuple__(),\
                    ctrl, self.state.__tuple__()))
            print("ERROR: Updated state makes agent out of bounds!!")

    def check_joint_state_safety(self, occupancy_dict=None, return_list=False):
        if occupancy_dict is None: occupancy_dict = self.supervisor.game.occupancy_dict
        is_safe = True
        for gridpt, agent in occupancy_dict.items():
            x, y, v = agent.state.x, agent.state.y, agent.state.v
            lead_agent = agent.find_lead_agent()
            # record all agents without valid back-up plan
            agents_no_bp = []
            if lead_agent:
                x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
                gap_curr = ((x_a-x)**2 + (y_a-y)**2)**0.5
                # not safe if gap is not large enough for any one of the agents
                if (self.compute_gap_req(lead_agent.a_min, v_a, self.a_min, v) > gap_curr):
                    is_safe = False
                    agents_no_bp.append(agent.state.__tuple__())
        
        if return_list: 
            return agents_no_bp
        else: 
            return is_safe

    def find_lead_agent(self, state=None, same_heading_required=True):
        if state is None:
            state = self.state
        try:
            arc_l, bundle = self.get_length_along_bundle()
        except:
            return None

        d_vec = DIRECTION_TO_VECTOR[state.heading]
        # get tiles in front
        tiles_x = np.arange(0,bundle.length-arc_l)*d_vec[0]+state.x
        tiles_y = np.arange(0,bundle.length-arc_l)*d_vec[1]+state.y

        # check agents in these tiles
        for i in range(0, len(tiles_x)):
            # as soon as agent found in nearest tile, return lead vehicle
            if (tiles_x[i], tiles_y[i]) in self.supervisor.game.occupancy_dict:
                agent = self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
                # only find agents that are facing in the same direction
                if (agent.get_id() != self.get_id()) and (not same_heading_required or (agent.state.heading == self.state.heading)):
                    return self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
        return None

    def compute_gap_req(self, lead_max_dec, lead_vel, follow_max_dec, follow_vel):
        #__import__('ipdb').set_trace(context=21)
        dx_lead = self.compute_dx(lead_max_dec, lead_vel)
        dx_behind = self.compute_dx(follow_max_dec, follow_vel)
        gap = max(dx_behind-dx_lead+1, 1)
        return gap
    

    # check if a set of actions is valid for a pair of agents
    def check_valid_actions(self, ag_1, ctrl_1, ag_2, ctrl_2):
        # get occupancy for both actions
        occ_1 = ag_1.query_occupancy(ctrl_1)
        occ_2 = ag_2.query_occupancy(ctrl_2)
        # if invalid actions, print an error
        if occ_1 is None or occ_2 is None:
            return False
        # check occupancy intersection
        chk_occupancy_intersection = self.check_occupancy_intersection(occ_1, occ_2)
        chk_safe_end_config = self.check_safe_config(ag_1, ag_2, occ_1[-1], occ_2[-1])
        # return if occupancies don't intersect and safe end config
        return not chk_occupancy_intersection and chk_safe_end_config

    def check_safe_config(self, ag_1, ag_2, st_1=None, st_2=None):
        if st_1 is None: st_1 = ag_1.state
        if st_2 is None: st_2 = ag_2.state

        # check agents are in the same lane
        def check_same_lane(ag_1, ag_2):
            try:
                width_1, bundle_1 = ag_1.get_width_along_bundle()
            except:
                return False
            try:
                width_2, bundle_2 = ag_2.get_width_along_bundle()
            except:
                return False
            return bundle_1.get_id() == bundle_2.get_id() and width_1 == width_2

        # returns agent_lead, agent_behind in that order
        def sort_agents(ag_1, ag_2):
            try:
                l_1 = ag_1.get_length_along_bundle()[0]
            except:
                return None, None, None, None
            try:
                l_2 = ag_2.get_length_along_bundle()[0]
            except:
                return None, None, None, None
            if l_1 > l_2:
                return ag_1, ag_2, st_1, st_2
            elif l_2 > l_1:
                return ag_2, ag_1, st_2, st_1
            else:
                return None, None, None, None

        # first check same lane
        same_lane_chk = check_same_lane(ag_1, ag_2)
        # TODO: if not in same lane, then agents are in safe config relative to each other?
        if not same_lane_chk: return True
        # then check which agent is lead and which one is behind
        ag_lead, ag_behind, st_lead, st_behind = sort_agents(ag_1, ag_2)
        # if None, agents are on top of each other
        if ag_lead is None: return False

        gap_req = self.compute_gap_req(ag_lead.a_min, st_lead.v, ag_behind.a_min, st_behind.v)
        gap_curr = np.linalg.norm(np.array([st_lead.x-st_behind.x, st_lead.y-st_behind.y]))
        return gap_curr >= gap_req

    def check_occupancy_intersection(self, occ_a, occ_b):
        # convert list of agent states to grid points if not already list of tuples
        if len(occ_a)>1: occ_a = occ_a[1:]
        if len(occ_b)>1: occ_b = occ_b[1:]
        if not isinstance(occ_a[0], tuple): occ_a = [(state.x, state.y) for state in occ_a]
        if not isinstance(occ_b[0], tuple): occ_b = [(state.x, state.y) for state in occ_b]
        occ_all = occ_a + occ_b
        if len(occ_all) != len(set(occ_all)):
            return True
        return False

    def get_backup_plan_ctrl(self, state=None):
        if state is None: state = self.state
        acc = self.a_min if state.v+self.a_min > 0 else -state.v
        return {'acceleration':acc, 'steer':'straight'}

    # checks if maximal yield action by receiver is enough...
    def intention_bp_conflict(self, agent):
        if agent.state.heading == self.state.heading: 
            chk_valid_actions = self.check_valid_actions(self, self.intention, agent, agent.get_backup_plan_ctrl())
            return not chk_valid_actions
        else:
            return False

    #=== helper methods for computing whether to send conflict request to another agent =====#
    def check_to_send_conflict_request(self, agent):
        # check if gap between two agents is large enough for stopping!
        # TODO: see whether end state after these actions still have a back-up plan
        def intentions_conflict(agent):
            if agent.state.heading == self.state.heading:
                chk_valid_actions = self.check_valid_actions(self, self.intention, agent, agent.intention)
                #if not chk_valid_actions:
                    #print("sending conflict request")
                return not chk_valid_actions
            else:
                return False
        
        # should only send conflict requests to agents facing the same direction
        return intentions_conflict(agent) #and self.state.heading == agent.state.heading 
        #or intention_forward_action_conflict(agent)

    #=== helper methods for computing the agent bubble ===================#
    def get_default_bubble(self, vel):
        # get all possible future reources car can occupy
        def get_resources():
            # set car state to have velocity v
            st = self.hack_state(self.state, x=self.default_state.x, y=self.default_state.y, heading=self.default_state.heading, v=vel)
            resources = []
            # make fake state here
            for ctrl in self.get_all_ctrl(state=st):
                #__import__('ipdb').set_trace(context=21)
                #print(ctrl)
                occ = self.query_occupancy(ctrl, state=st, inverse=False)
                #[print(state.x, state.y) for state in occ]
                if occ is not None:
                    resources_to_add = [(state.x, state.y) for state in occ]
                    resources.extend(resources_to_add)
                    #print(resources)
                    # add in elements where car might need to execute emergency braking from the final state
                    final_st = occ[-1]
                    state_to_chk = self.hack_state(self.state, x=final_st.x, \
                        y=final_st.y, heading=final_st.heading, v=final_st.v)
                    safety_plan_resources = self.get_tiles_for_safety_plan(state=state_to_chk)
                    resources.extend(safety_plan_resources)
            return resources

        # gridpoints
        bubble = []
        resources = get_resources()
        # get all possible reachable states
        for xy in resources:
            # get all possible states to that resource
            # TODO: Right now, assuming all other agents have the same dynamics as itself!!
            states = self.get_backwards_reachable_states_from_gridpoint(xy)
            gridpts = [(state.x, state.y) for state in states]
            bubble.extend(gridpts)
                
        # plot the bubble
        '''fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        for grid in bubble:
            rect = patches.Rectangle((grid[1],grid[0]),1,1,linewidth=0.5,facecolor='grey', alpha=0.3)
            ax.add_patch(rect)
        plt.show()'''

        return list(set(bubble))

    # compute number of tiles when applying brakes maximally
    def compute_dx(self, a_min, vel):
        dt = math.ceil(-vel/a_min)-1
        dx = int(np.sum([vel+(k+1)*a_min for k in range(dt)]))
        return dx

    # returns the tiles used in executing safety plan
    # note need agent to get max dec
    def get_tiles_for_safety_plan(self, state):
        dx = self.compute_dx(self.a_min, state.v)
        # populate tiles in front according to that depth for east facing agent
        gridpts = [(0, j+1) for j in range(dx)]
        if len(gridpts) == 0: gridpts.append((0,0))
        # rotate tiles to local agent frame
        dx, dy, dtheta = self.find_transformation(self.default_state.__tuple__(), state.__tuple__())
        gridpts_tf = [self.transform_state(gridpt, dx, dy, dtheta) for gridpt in gridpts]
        return gridpts_tf

    # getting state from given final state and inverse control action
    def compute_state_from(self, state_to, inv_ctrl):
        def get_prev_heading(heading, act):
            heading = Car.convert_orientation(heading)
            if act == 'right-turn':
                heading += 90
            elif act == 'left-turn':
                heading -= 90
            return Car.convert_orientation(heading)

        acc = inv_ctrl['acceleration']
        steer = inv_ctrl['steer']
        # if previous state has velocity out of bounds, invalid state_from
        if state_to.v-acc < self.v_min or state_to.v-acc > self.v_max:
            return None
        v_0 = state_to.v-acc
        phi_0 = get_prev_heading(state_to.heading, steer)

        # simulate forward to get the appropriate dx and dy
        pseudo_state = self.hack_state(self.state, x=state_to.x, y=state_to.y, heading=phi_0, v=v_0)
        state_from = self.query_occupancy(ctrl=inv_ctrl, state=pseudo_state, inverse=True)
        if state_from is None: return None

        # the first element of the occupancy_list is where the agent came from
        return state_from[0]

    # compute all backwards reachable states and actions to get to state specified
    def get_backwards_reachable_states(self, state_to):
        # just need a car object to call get_all_ctrl function
        all_inv_ctrl = self.get_all_ctrl(state=state_to, inverse=True)
        states = [None]*len(all_inv_ctrl) # will be too large but saves time pre-allocating
        actions = [None]*len(all_inv_ctrl)
        cnt = 0
        for inv_ctrl in all_inv_ctrl:
            occupancy = self.compute_state_from(state_to, inv_ctrl)
            if occupancy is not None:
                states[cnt] = occupancy
                actions[cnt] = inv_ctrl
                cnt = cnt+1
        # remove all the none
        states = states[0:cnt]
        actions = actions[0:cnt]
        return states, actions

    # get all backwards reachable states from a gridpoint
    def get_backwards_reachable_states_from_gridpoint(self, xy, filter_dir=None):
        directions = list(DIRECTION_TO_VECTOR.keys())
        vel_arr = np.arange(self.v_min, self.v_max+1)
        backwards_reachable_states = []
        for dir in directions:
            for vel in vel_arr:
                # create a pseudo state
                state_to = self.hack_state(self.state, x=xy[0], y=xy[1], heading=dir, v=vel)
                states, _ = self.get_backwards_reachable_states(state_to)
                # filter for cars coming from certain direction
                if filter_dir is not None:
                    states = [state for state in states if state.heading == filter_dir]
                backwards_reachable_states.extend(states)
        return backwards_reachable_states

    # transform the state with rotation and translation, do not assign heading
    # can be for tuple coordinate (x, y) or tuple state (x,y, heading, v)
    def transform_state(self, state, dx, dy, dtheta, assign_heading=False):
        # rotate and translate the vector
        arr = rotate_vector([state[0],state[1]], dtheta*np.pi/180)
        if assign_heading:
            return (arr[0]+dx, arr[1]+dy, \
                Car.convert_orientation(Car.convert_orientation(state[2])+dtheta), state[3])
        else:
            return (arr[0]+dx, arr[1]+dy)

    # find transformation between initial state and final state
    def find_transformation(self, init_state, final_state):
        dx = final_state[0]-init_state[0]
        dy = final_state[1]-init_state[1]
        dtheta = Car.convert_orientation(final_state[2]) - Car.convert_orientation(init_state[2])
        return dx, dy, dtheta


class Artist:
    def __init__(self, stdscr, start_point, max_yx):
        self.stdscr = stdscr
        self.x_min = start_point[0]
        self.y_min = start_point[1]
        self.x_max = max_yx[0]
        self.y_max = max_yx[1]

    def draw_set(self, set_of_drawable_objects):
        for drawable_object in set_of_drawable_objects:
            drawables = drawable_object.get_drawables()
            for drawable in drawables:
                x,y = drawable.xy
                self.draw(x, y, drawable.drawstr)

    def enable_window_moving(self):
        key = self.stdscr.getch()
        if key == curses.KEY_RIGHT:
            self.y_min += 1
            self.y_max += 1
        elif key == curses.KEY_LEFT:
            self.y_min -= 1
            self.y_max -= 1
        elif key == curses.KEY_UP:
            self.x_min -= 1
            self.x_max -= 1
        elif key == curses.KEY_DOWN:
            self.x_min += 1
            self.x_max += 1

    def draw(self, x, y, draw_str):
        def in_range(a, a_min, a_max):
            return a >= a_min and a < a_max

        if in_range(x, self.x_min, self.x_max-1) and in_range(y, self.y_min, self.y_max):
            self.stdscr.addstr(x-self.x_min, y-self.y_min, draw_str)

# verifies whether or not spawned agent passes certain safety checks
class SpawningContract():
    def __init__(self, game, new_agent):
        self.game = game
        self.new_agent = new_agent
        self.okay_to_spawn_flag = self.passes_all_checks()

    # passes all necessary checks 
    def passes_all_checks(self):
        #print(self.valid_init_state(), self.valid_init_safe_state(), \
        #    self.valid_traffic_state_for_traffic_lights(), self.agent_not_in_intersection(), self.agent_facing_right_direction())
        all_checks = self.valid_init_state() and self.valid_init_safe_state() and \
            self.valid_traffic_state_for_traffic_lights() and self.agent_not_in_intersection() \
                and self.agent_facing_right_direction()
        return all_checks

    # check to make sure not overlapping with other agents
    def valid_init_state(self):
        return (self.new_agent.state.x, self.new_agent.state.y) not in self.game.occupancy_dict 

    # check that not in inevitable collision state with other agents or static obstacles
    def valid_init_safe_state(self):
        # make sure that max back-up plan can avoid collision with wall
        valid_static_obs = True
        tiles = self.new_agent.get_tiles_for_safety_plan(self.new_agent.state)
        for tile in tiles: 
            if tile not in self.game.map.drivable_nodes:
                valid_static_obs = False

        # make sure that joint state of all agents with this agent included is safe
        # add to a copy of the occupancy dict
        valid_joint_state = True
        occupancy_dict_chk = self.game.occupancy_dict.copy()
        occupancy_dict_chk[(self.new_agent.state.x, self.new_agent.state.y)] = self.new_agent
        valid_joint_state = self.new_agent.check_joint_state_safety(occupancy_dict=occupancy_dict_chk)
        return valid_joint_state and valid_static_obs

    # check to make sure agent isn't in a state that makes it impossible to follow
    # traffic rules
    def valid_traffic_state_for_traffic_lights(self):
        traffic_light_oracle = TrafficLightOracle()
        for ctrl in self.new_agent.get_all_ctrl():
            chk_ctrl = traffic_light_oracle.evaluate(ctrl, self.new_agent, self.game)
            if chk_ctrl: return True
        return False

    # check to make sure agent is not in an intersection
    def agent_not_in_intersection(self):
        orientations = self.game.map.legal_orientations[(self.new_agent.state.x, self.new_agent.state.y)]
        return not (len(orientations) > 1)
    
    # check agent is in the right legal orientation
    def agent_facing_right_direction(self):
        orientation = self.game.map.legal_orientations[(self.new_agent.state.x, self.new_agent.state.y)]
        return orientation[0] == self.new_agent.state.heading

class Game:
    # combines scenario + agents for game
    def __init__(self, game_map, agent_set=[]):
        self.time = 0
        self.map = game_map
        self.agent_set = agent_set
        self.draw_sets = [self.map.drivable_tiles, self.map.traffic_lights, self.agent_set] # list ordering determines draw ordering
        self.traces = {"agent_ids":[]}
        self.occupancy_dict = dict()
        self.update_occupancy_dict()
        self.collision_dict = dict()
        self.out_of_bounds_dict = dict()
        self.unsafe_joint_state_dict = dict()

    def update_occupancy_dict(self):
        occupancy_dict = dict()
        for agent in self.agent_set:
            x, y = agent.state.x, agent.state.y
            occupancy_dict[x,y] = agent
        self.occupancy_dict = occupancy_dict

    def spawn_agents(self):
        def valid_source_sink(source, sink):
            return not (source.node[0] == sink.node[0] or source.node[1] == sink.node[1])

        for source in self.map.IO_map.sources:
            if np.random.uniform() <= source.p:
                sink = np.random.choice(self.map.IO_map.map[source])
                if not valid_source_sink(source, sink):
                    return

                # check if new car satisfies spawning safety contract
                new_car = create_default_car(source, sink, self)
                spawning_contract = SpawningContract(self, new_car)
                if spawning_contract.okay_to_spawn_flag:
                    self.agent_set.append(new_car)
                    self.update_occupancy_dict()

    def add_agent(self, agent):
        self.agent_set.append(agent)
    def time_forward(self):
        self.time += 1
    def save_plotting_info(self):
        lights = []
        agents = []
        # save all the agent states
        for agent in self.agent_set:
            spec_struct_trace = {}
            if self.time > 0:
                spec_struct_trace = agent.spec_struct_trace
            agents.append((agent.state.x, agent.state.y, \
                agent.state.heading, agent.state.v, agent.agent_color, agent.get_bubble(), agent.get_id()))
        # save all the traffic light states
        for traffic_light in self.map.traffic_lights:
            for tile in traffic_light.htiles:
                x, y = tile.xy
                lights.append((x, y, traffic_light.get_hstate()[0], traffic_light.htimer, 'horizontal', traffic_light.durations))
            for tile in traffic_light.vtiles:
                x, y = tile.xy
                lights.append((x, y, traffic_light.get_vstate()[0], traffic_light.htimer, 'vertical', traffic_light.durations))

        # return dict with all the info
        self.traces[self.time] = {"lights": lights, "agents": agents}
    
    def write_agents_to_traces(self):
        for agent in self.agent_set:
            # unpack agents in bubble to tuples
            agents_in_bubble = [[agent.state.__tuple__(), agent.get_id()]  for agent in agent.agents_in_bubble]
            # unpack agents in send and receive requests to tuples
            sent = [agent.state.__tuple__() for agent in agent.send_conflict_requests_to]
            received = [agent.state.__tuple__() for agent in agent.received_conflict_requests_from] 
            max_not_braking_enough = agent.agent_max_braking_not_enough
            agent_intention = agent.intention
            if agent.intention is not None: 
                agent_intention = (agent.intention['acceleration'], agent.intention['steer'])
            if max_not_braking_enough is not None:
                max_not_braking_enough = agent.agent_max_braking_not_enough.state.__tuple__()
            # save all data in trace
            agent_trace_dict = {'state':(agent.state.x, agent.state.y, agent.state.heading, agent.state.v), 'action': agent.ctrl_chosen, \
                'color':agent.agent_color, 'bubble':agent.get_bubble(), 'goals': agent.supervisor.goals, \
                    'spec_struct_info': agent.spec_struct_trace, 'agents_in_bubble': agents_in_bubble, \
                        'sent_request': sent, 'received_requests': received, 'token_count':agent.token_count, \
                            'max_braking_not_enough': max_not_braking_enough, \
                                'straight_action_eval': agent.straight_action_eval, \
                                    'action_selection_flags': agent.action_selection_flags, \
                                        'intention': agent_intention}

            # if not yet in traces, add it to traces
            agent_id = agent.get_id()
            if agent_id not in self.traces:
                self.traces["agent_ids"].append(agent_id)
                agent_param = {"v_min":agent.v_min, "v_max":agent.v_max, "a_min":agent.a_min, "a_max":agent.a_max}
                self.traces[agent_id] = {'agent_param':agent_param, self.time:agent_trace_dict}
            # else update its dict 
            self.traces[agent_id][self.time] = agent_trace_dict
    
    
    # write the game information to traces
    def write_game_info_to_traces(self, t_end):
        self.traces["map_name"] = self.map.map_name
        self.traces["spawn_probability"] = self.map.default_spawn_probability
        self.traces["collision_dict"] = self.collision_dict
        self.traces["out_of_bounds_dict"] = self.out_of_bounds_dict
        self.traces["unsafe_joint_state_dict"] = self.unsafe_joint_state_dict
        self.traces["t_end"] = t_end

    def write_data_to_pckl(self, filename, traces, new_entry=None):
        if new_entry is not None:
            traces.update(new_entry)
        with open(filename, 'wb+') as pckl_file:
            pickle.dump(traces, pckl_file)

    # outfile flag is if you want to save data to file, it should be in pickle format
    def animate(self, frequency, t_end=np.inf):
        stdscr = curses.initscr()
        height,width = stdscr.getmaxyx()
        artist = Artist(stdscr, [0, 0], [height, width])
        try:
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            stdscr.keypad(True)
            stdscr.nodelay(True)

            while self.time < t_end:
                artist.enable_window_moving()

                stdscr.clear()
                for node in self.map.nodes:
                    node_x, node_y = node
                    if self.map.grid[node] == '-':
                        draw_str = '.'
                    else:
                        draw_str = self.map.grid[node]
                    artist.draw(node_x, node_y, draw_str)

                for traffic_light in self.map.traffic_lights:
                    drawables = traffic_light.get_drawables()
                    for drawable in drawables:
                        x, y = drawable.xy
                        artist.draw(x, y, drawable.drawstr)

                for agent in self.agent_set:
                    artist.draw(agent.state.x, agent.state.y, agent.get_symbol())

                stdscr.clear()
                # update states
                self.play_step()
                # draw objects
                for draw_set in self.draw_sets:
                    artist.draw_set(draw_set)
                stdscr.refresh()
                time.sleep(frequency)
                self.time_forward()
        finally:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()

    def sys_step(self):
        run(self.agent_set)

    def env_step(self):
        run(self.map.traffic_lights)
        self.spawn_agents()

    def play_step(self):
        self.sys_step()
        self.env_step()

    # code for fixing the agent states
    def fix_agent_states_karena_debug(self, states, intentions=None):
        #print(len(states), len(intentions))
        for i, agent in enumerate(self.agent_set):
            st = agent.hack_state(agent.state, x=states[i][0], y=states[i][1], heading=states[i][2], v=states[i][3])
            agent.state = st
            if intentions is not None:
                agent.set_debug_intention(intentions[i])
        self.update_occupancy_dict()

    # code for debugging agent conflict requests
    def check_conflict_requests_karena_debug(self):
        #states = [(8,1,'east',1), (8,3,'east',1), (8,4,'east',1), (9,2,'east',1), (9,3,'east',1)]
        #intentions = [{'acceleration': 0, 'steer': 'left-lane' }, {'acceleration': 0, 'steer':'right-lane'}, \
        #    {'acceleration': 0, 'steer': 'right-lane' }, {'acceleration': 0, 'steer': 'straight' }, {'acceleration': 0, 'steer': 'left-lane'}]
        states = [(9,0, 'east',0), (10,0,'east',0)]
        intentions = [{'acceleration': 1, 'steer': 'right-lane'}, {'acceleration': 2, 'steer': 'left-lane' }]
        self.fix_agent_states_karena_debug(states, intentions)
        # checking conflict swapping code
        self.send_and_receive_conflict_requests()
        for agent in self.agent_set:
            print("NEW AGENT")
            print(agent.state.__tuple__())
            ag = agent.agent_max_braking_not_enough
            if ag is not None:
                print("maximal yielding flag")
                print(agent.agent_max_braking_not_enough.state.__tuple__())
            print("send conflict requests to")
            for ag in agent.send_conflict_requests_to:
                print(ag.state.__tuple__())
            print("received conflict requests from")
            for ag in agent.received_conflict_requests_from:
                print(ag.state.__tuple__())

    # check that all agents in the current config have a backup plan
    def check_config_safety(self):
        for agent in agent_set:
            x, y, v = agent.state.x, agent.state.y, agent.state.v
            lead_agent = plant.find_lead_agent()
            if lead_agent:
                x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
                gap_curr = ((x_a-x)**2 + (y_a-y)**2)**0.5
                # not safe if gap is not large enough for any one of the agents
                if (plant.compute_gap_req(lead_agent.a_min, v_a, plant.a_min, v) >= gap_curr):
                    return False
        # all agents have backup plan
        return True

    # loop through all agents and send and receive conflict requests
    def send_and_receive_conflict_requests(self):
        # reset all requests
        for agent in self.agent_set:
            agent.reset_conflict_lists()

        for agent in self.agent_set:
            # set list to send requests to
            agent.send_conflict_requests_to = agent.find_agents_to_send_conflict_request()

            # for each agent receiving request, update their receive list
            for agent_rec in agent.send_conflict_requests_to:
                agent_rec.received_conflict_requests_from.append(agent)

    def play(self, t_end=np.inf, outfile=None):
        # dump the map here and open json file
        write_bool = outfile is not None and t_end is not np.inf
        while self.time < t_end:
            print("TIME: " + str(self.time))
            # if save data to animate
            if write_bool:
                self.write_agents_to_traces()
                self.save_plotting_info()

            self.play_step()
            self.time_forward()
        

        if write_bool:
            output_dir = os.getcwd()+'/saved_traces/'
            self.write_game_info_to_traces(t_end)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.write_data_to_pckl(output_dir + outfile, self.traces)

def symbol_to_orientation(symb):
    if symb in ['←','⇠']:
        orientation = 'west'
    elif symb in ['→','⇢']:
        orientation = 'east'
    elif symb in ['↓','⇣']:
        orientation = 'south'
    elif symb in ['↑','⇡']:
        orientation = 'north'
    else:
        orientation = None
    return orientation

class DrivableTile:
    def __init__(self, xy, symbol):
        self.xy = xy
        self.symbol = symbol
    def get_symbol(self):
        return self.symbol
    def get_drawables(self):
        drawables = []
        drawable = Drawable(self.xy, self.get_symbol())
        drawables.append(drawable)
        return drawables

class Bundle:
    def __init__(self, tube_list, length_list, direction):
        self.direction = direction
        self.tube_list = self.get_width_list(tube_list)
        self.length_list = self.get_length_list(length_list)
        self.tiles = self.reconstruct_tiles()
        self.length = len(self.length_list)
        self.width = len(self.tube_list)

    def get_id(self):
        return id(self)

    def relative_coordinates_to_tile(self, rel_xy):
        x, y = rel_xy
        dvec = DIRECTION_TO_VECTOR[self.direction]
        tube_idx = 1-np.nonzero(DIRECTION_TO_VECTOR[self.direction])[0][0]
        tile = [None] * 2
        tile[tube_idx] = self.tube_list[x]
        tile[1-tube_idx] = self.length_list[y]
        return tuple(tile)

    def reconstruct_tiles(self):
        dvec = DIRECTION_TO_VECTOR[self.direction]
        tube_idx = 1-np.nonzero(DIRECTION_TO_VECTOR[self.direction])[0][0]
        tiles = []
        for x in range(len(self.tube_list)):
            for y in range(len(self.length_list)):
                tile = self.relative_coordinates_to_tile((x,y))
                tiles.append(tile)
        return tiles

    def get_length_list(self, length_list):
        increasing_dir = DIRECTION_TO_VECTOR[self.direction][np.nonzero(DIRECTION_TO_VECTOR[self.direction])[0][0]]
        reverse = True
        if increasing_dir > 0:
            reverse = False
        length_list.sort(reverse=reverse)
        return length_list

    def get_width_list(self, tube_list):
        reverse = True
        if self.direction == 'west' or self.direction == 'south':
            reverse = False
        tube_list.sort(reverse=reverse)
        return tube_list

    def tile_to_relative_length(self, tile):
        stp_idx = np.nonzero(DIRECTION_TO_VECTOR[self.direction])[0][0]
        return self.length_list.index(tile[stp_idx])

    def tile_to_relative_width(self, tile):
        tube_idx = 1-np.nonzero(DIRECTION_TO_VECTOR[self.direction])[0][0]
        return self.tube_list.index(tile[tube_idx])

    def tile_to_relative_position(self, tile):
        width = self.tile_to_relative_width(tile)
        length = self.tile_to_relative_length(tile)
        return width, length

    def is_leftmost_lane(self, tile):
        return self.tile_to_relative_width(tile) == self.width - 1

    def is_rightmost_lane(self, tile):
        return self.tile_to_relative_width(tile) == 0

def get_comparator(cond):
    if cond == 'equal':
        return lambda curr, nxt: nxt is curr
    elif cond == 'increasing':
        return lambda curr, nxt: nxt == curr + 1
    elif cond == 'decreasing':
        return lambda curr, nxt: nxt == curr - 1

def separate_list(lst, cond, sort=False):
    if sort:
        lst.sort()
    partitions = []
    part = [lst[0]]
    comparator = get_comparator(cond)
    for curr, nxt in zip(lst, lst[1:]):
        if comparator(curr, nxt):
            part.append(nxt)
        else:
            partitions.append(part)
            part = [nxt]
    partitions.append(part)
    return partitions

def append_or_create_new_list(dictionary, key, item):
    if key in dictionary:
        dictionary[key].append(item)
    else:
        dictionary[key] = [item]

class Map:
    def __init__(self, csv_filename, default_spawn_probability=0.9, random_traffic_lights_init=False):
        self.map_name = csv_filename
        self.grid = self.get_grid(csv_filename)
        self.default_spawn_probability = default_spawn_probability
        self.nodes = list(self.grid.keys())
        self.drivable_tiles, self.drivable_nodes, self.non_drivable_nodes = self.get_drivable_tiles()
        self.legal_orientations = self.get_legal_orientations()
        self.road_map = self.get_road_map()
        self.intersections = self.get_intersections()
        self.traffic_lights = self.get_traffic_lights(random_traffic_lights_init)
        self.intersection_to_traffic_light_map = self.get_intersection_to_traffic_light_map()
        self.tile_to_intersection_map = self.get_tile_to_intersection_map()
        self.bundles = self.get_bundles()
        self.tile_to_bundle_map = self.get_tile_to_bundle_map()
        self.IO_map = self.get_IO_map()
        self.traffic_light_tile_to_bundle_map = self.get_traffic_light_tile_to_bundle_map()
        self.tile_to_traffic_light_map = self.get_tile_to_traffic_light_map()
        self.special_goal_tiles = []
        self.right_turn_tiles = self.find_right_turn_tiles()
        self.left_turn_tiles = self.find_left_turn_tiles()
        self.all_left_turns = self.get_all_left_turns()
        self.bundle_graph = self.get_bundle_graph()
        self.left_turn_to_opposing_traffic_bundles = self.get_left_turn_to_opposing_traffic_map()

    # for now, we are assuming default car dynamics; TODO: generalize this
    def get_left_turn_to_opposing_traffic_map(self):
        left_turn_to_opposing_traffic_map = dict()
        for bundle in self.left_turn_tiles:
            for left_turn_tile in self.left_turn_tiles[bundle]:
                next_tiles = self.left_turn_tiles[bundle][left_turn_tile]
                first_xy, first_heading = next_tiles[0]
                second_xy, second_heading = next_tiles[1]
                opposing_bundle_heading = Car.convert_orientation(Car.convert_orientation(first_heading) + 90)
                opposing_bundle = self.directed_tile_to_bundle((first_xy, opposing_bundle_heading))
                first_rel_x = opposing_bundle.tile_to_relative_length(first_xy)
                first_rel_y = opposing_bundle.tile_to_relative_width(first_xy)
                second_rel_x = opposing_bundle.tile_to_relative_length(second_xy)
                relative_occupancy = [second_rel_x] * (first_rel_y + 1)
                relative_occupancy[-1] = first_rel_x
                relative_occupancy[-2] = second_rel_x

                left_turn_to_opposing_traffic_map[left_turn_tile] = opposing_bundle, relative_occupancy
        return left_turn_to_opposing_traffic_map

    def change_traffic_lights(self, traffic_tile, hstate, htimer):
        traffic_light = self.tile_to_traffic_light_map[traffic_tile]
        assert hstate in ['red', 'yellow', 'green']
        assert htimer < traffic_light.durations[hstate]
        traffic_light.hstate = hstate
        traffic_light.htimer = htimer

    def get_bundle_plan(self, source, sink):
        planning_graph = self.bundle_graph.copy()
        original_edges = list(planning_graph.edges)

        for edge in original_edges:
            end_node = edge[-1]
            if self.check_directed_tile_reachability(end_node, sink):
                planning_graph.add_edge(end_node, sink)

        if self.check_directed_tile_reachability(source, sink):
            planning_graph.add_edge(source, sink)

        turns = self.directed_tile_to_turns(source)
        for turn in turns:
            planning_graph.add_edge(source, turn)

        plan = nx.astar_path(planning_graph, source, sink)
        return plan

    def directed_tile_to_relative_bundle_tile(self, directed_tile):
        tile, _ = directed_tile
        bundle = self.directed_tile_to_bundle(directed_tile)
        rel_tile = bundle.tile_to_relative_position(tile)
        return rel_tile

    def check_directed_tile_reachability(self, dtile_start, dtile_final):
        """
        check if there exists a forward-only sequence of control actions that will
        take a car from a starting directed tile to a final directed tile, this is
        assuming the default car dynamics.
        """
        # check if dtile_start is a list of directed tiles, if so take the last one
        if not isinstance(dtile_start[1], str):
            dtile_start = dtile_start[-1]
        # check if dtile_final is a list of directed tiles, if so take the first one
        if not isinstance(dtile_final[1], str):
            dtile_final = dtile_final[0]

        # determine which bundle dtile_start is in
        bundle_start = self.directed_tile_to_bundle(dtile_start)
        # determine which bundle dtile_final is in
        bundle_final = self.directed_tile_to_bundle(dtile_final)

        # if the two directed tiles do not belong to the same bundle
        if bundle_start != bundle_final:
            return False
        # else, we do a tile count and check if the longitudinal difference is
        # at least equal to the lateral difference
        else:
            rel_tile_start = self.directed_tile_to_relative_bundle_tile(dtile_start)
            rel_tile_final = self.directed_tile_to_relative_bundle_tile(dtile_final)
            length_diff = rel_tile_final[1] - rel_tile_start[1]
            width_diff = abs(rel_tile_final[0]- rel_tile_start[0])
            return length_diff >= width_diff

    def directed_tile_to_turns(self, directed_tile):
        """
        compute (first) critical points for turning along bundles
        """
        # list of critical turning tiles
        turns = []
        bundle = self.directed_tile_to_bundle(directed_tile)

        # process right turns
        for turn in self.right_turn_tiles[bundle]:
            if self.check_directed_tile_reachability(directed_tile, turn):
                turns.append(turn)

        # process left turns
        for turn in self.left_turn_tiles[bundle]:
            precrossing_tile = self.get_precrossing_left_turn_tile(turn)
            stop_tiles = self.left_turn_tiles[bundle][turn]
            if precrossing_tile is None:
                turn_node = turn
                precrossing_tile = (turn,) + stop_tiles
            else:
                turn_node = (precrossing_tile, turn) + stop_tiles
            if self.check_directed_tile_reachability(directed_tile, precrossing_tile):
                turns.append(turn_node[0:-1])
        return turns

    def get_bundle_graph(self):
        '''
        constructs bundle graph
        '''
        bundle_graph = nx.DiGraph()
        # process right turns
        for bundle in self.right_turn_tiles:
            for from_tile in self.right_turn_tiles[bundle]:
                to_tiles = self.right_turn_tiles[bundle][from_tile]
                # get last tile
                to_tile = to_tiles[-1]
                # add edge from critical tile to last configuration
                bundle_graph.add_edge(from_tile, to_tile)
                # get critical tiles reachable from to_tile
                turns = self.directed_tile_to_turns(to_tile)
                for turn in turns:
                    # add the corresponding directed edges
                    bundle_graph.add_edge(to_tile, turn)

        # process left turns
        for bundle in self.left_turn_tiles:
            for from_tile in self.left_turn_tiles[bundle]:
                to_tiles = self.left_turn_tiles[bundle][from_tile]
                # check if a procrossing_tile is required
                precrossing_tile = self.get_precrossing_left_turn_tile(from_tile)
                if precrossing_tile:
                    frm = (precrossing_tile, from_tile) + to_tiles[0:-1]
                    to = to_tiles[-1]
                    bundle_graph.add_edge(frm, to)
                else:
                    frm = (from_tile,) + to_tiles[0:-1]
                    to = to_tiles[-1]
                    bundle_graph.add_edge(frm, to)
                # get critical tiles reachable from last tile in to_tiles
                turns = self.directed_tile_to_turns(to_tiles[-1])
                for turn in turns:
                    bundle_graph.add_edge(to_tiles[-1], turn)

        return bundle_graph

    def get_precrossing_left_turn_tile(self, left_turn_tile):
        """
        given a turn tile, find if it exists its closest legal predecessor
        before entering an intersection
        """
        tile_xy, tile_direction = left_turn_tile
        # check if turn tile is not in an intersection
        if len(self.legal_orientations[tile_xy]) <= 1:
            return None
        else:
            # keep going backward until no longer in the intersection
            backward = -np.array(DIRECTION_TO_VECTOR[tile_direction])
            new_tile = tuple(np.array(tile_xy)+backward)
            while True:
                if new_tile not in self.legal_orientations or self.legal_orientations[new_tile] is None:
                    return None
                elif len(self.legal_orientations[new_tile]) > 1:
                    new_tile = tuple(np.array(new_tile) + backward)
                else:
                    return new_tile, tile_direction

    def check_if_right_turn_tile(self, directed_tile):
        tile, direction = directed_tile
        try:
            assert direction in self.legal_orientations[tile]
        except:
            return False, None
        direction_degrees = Car.convert_orientation(direction)
        next_direction_degrees = (direction_degrees - 90)%360
        next_direction = Car.convert_orientation(next_direction_degrees)
        forward = DIRECTION_TO_VECTOR[direction]
        right = rotate_vector(forward, -np.pi/2)
        next_tile = tuple(np.array(tile) + np.array(forward) + np.array(right))
        try:
            next_bundle = self.directed_tile_to_bundle((next_tile, next_direction))
            return next_bundle.is_rightmost_lane(next_tile), ((next_tile, next_direction),)
        except:
            return False, None

    def check_if_left_turn_type_A(self, directed_tile):
        """
        check if left turn tile is of type A (from two-way street into two-way
        street), assuming directed tile input is legal
        """
        tile, direction = directed_tile
        direction_degrees = Car.convert_orientation(direction)
        next_direction_degrees = (direction_degrees + 90) % 360
        next_direction = Car.convert_orientation(next_direction_degrees)

        forward = DIRECTION_TO_VECTOR[direction]
        left = rotate_vector(forward, np.pi/2)
        next_tile = tuple(np.array(tile) + np.array(forward) + np.array(left))
        first_stop = (next_tile, next_direction)

        next_next_tile = tuple(np.array(next_tile) + np.array(left) + np.array(forward))
        second_stop = (next_next_tile, next_direction)
        try:
            next_bundle = self.directed_tile_to_bundle(second_stop)
            if next_bundle.is_leftmost_lane(next_next_tile):
                # update legal orientation dictionary
                self.legal_orientations[next_tile].append(next_direction)
                # add to special tiles the first stop
                self.special_goal_tiles.append(first_stop)
                return True, (first_stop, second_stop)
            else:
                return False, None
        except:
            return False, None

    def check_if_left_turn_type_B(self, directed_tile):
        """
        check if left turn tile is of type B (direct turn into one-way street),
        assuming directed tile input is legal;
        this turn can only be made from a non-intersection tile
        """
        tile, direction = directed_tile
        if len(self.legal_orientations[tile]) > 1:
            return False, None
        else:
            direction_degrees = Car.convert_orientation(direction)
            next_direction_degrees = (direction_degrees + 90) % 360
            next_direction = Car.convert_orientation(next_direction_degrees)

            forward = DIRECTION_TO_VECTOR[direction]
            left = rotate_vector(forward, np.pi/2)
            next_tile = tuple(np.array(tile) + np.array(forward) + np.array(left))
            next_stop = (next_tile, next_direction)
            try:
                next_bundle = self.directed_tile_to_bundle((next_tile, next_direction))
                if next_bundle.is_leftmost_lane(next_tile):
                    return True, (next_stop,)
                else: return False, None
            except:
                return False, None

    def check_if_left_turn_tile(self, directed_tile):
        """
        given a tile, check if it is ok to perform a left-turn from there

        """
        tile, direction = directed_tile
        # check if tile is even legal
        try:
            assert direction in self.legal_orientations[tile]
        except:
            return False, None
        type_A_check = self.check_if_left_turn_type_A(directed_tile)
        if type_A_check[0]:
            return type_A_check
        else:
            type_B_check = self.check_if_left_turn_type_B(directed_tile)
            if type_B_check[0]:
                return type_B_check
            else:
                return False, None

    # assuming agents can only legally make a right turn from the rightmost lane into rightmost lane
    def find_right_turn_tiles(self):
        """
        collect all (final) right-turn tiles in the map using the check function
        """
        right_turn_tiles = dict()
        for bundle in self.bundles:
            right_turn_tiles[bundle] = dict()
            direction = bundle.direction
            for idx in range(bundle.length):
                tile = bundle.relative_coordinates_to_tile((0, idx))
                directed_tile = (tile, direction)
                check, nxt = self.check_if_right_turn_tile(directed_tile)
                if check:
                    right_turn_tiles[bundle][(tile, direction)] = nxt
        return right_turn_tiles

    # assuming agents can only legally make a left turn from the leftmost lane into leftmost lane
    def find_left_turn_tiles(self):
        """
        collect all (final) left-turn tiles in the map using the check function
        """

        left_turn_tiles = dict()
        # goes through each bundle
        for bundle in self.bundles:
            # create a dictionary entry for the bundle
            left_turn_tiles[bundle] = dict()
            direction = bundle.direction
            for idx in range(bundle.length):
                # pick tile in left-most lane
                tile = bundle.relative_coordinates_to_tile((bundle.width-1, idx))
                directed_tile = (tile, direction)
                # performs the check
                check, nxt = self.check_if_left_turn_tile(directed_tile)
                if check:
                    # if check succeeds, add directed tile as a key to the
                    # dictionary corresponding to the bundle with the
                    # corresponding value nxt being the remaining tiles
                    # required to complete the left-turn
                    left_turn_tiles[bundle][(tile, direction)] = nxt
        return left_turn_tiles

    def get_all_left_turns(self):
        all_left_turns = []
        for bundle in self.left_turn_tiles:
            for left_turn in self.left_turn_tiles[bundle]:
                all_left_turns.append(left_turn)
        return all_left_turns

    def directed_tile_to_bundle(self, directed_tile):
        """
        given a tile and a direction, return the bundle containing that tile
        """
        # TODO: Fix to not have heading=None
        # assert tile in self.tile_to_bundle_map, 'Tile does not belong to any bundle!'
        tile, heading = directed_tile
        if tile not in self.tile_to_bundle_map:
            return None
        bundles = self.tile_to_bundle_map[tile]
        for bundle in bundles:
            if heading == bundle.direction:
                return bundle
        return None
        # try:
        #bundle_bool = np.nonzero([b.direction == heading for b in bundles])
        #if len(bundle_bool) == 0:
        #    return None
        #else:
        #    bundle_idx = bundle_bool[0][0]
        #    bundle = bundles[bundle_idx]

    def get_tile_to_traffic_light_map(self):
        tile_to_traffic_light_map = dict()
        for traffic_light in self.traffic_lights:
            for tile in traffic_light.vtiles:
                tile_to_traffic_light_map[tile.xy] = traffic_light
            for tile in traffic_light.htiles:
                tile_to_traffic_light_map[tile.xy] = traffic_light
        return tile_to_traffic_light_map

    def get_traffic_light_tile_to_bundle_map(self):
        tile_to_bundle_map = dict()
        for traffic_light in self.traffic_lights:
            for tile in traffic_light.vtiles:
                bundle = self.tile_to_bundle_map[tile.xy]
                append_or_create_new_list(tile_to_bundle_map, tile.xy, bundle)
            for tile in traffic_light.htiles:
                bundle = self.tile_to_bundle_map[tile.xy]
                append_or_create_new_list(tile_to_bundle_map, tile.xy, bundle)
        return tile_to_bundle_map

    def get_tile_to_bundle_map(self):
        tile_to_bundle_map = dict()
        for bundle in self.bundles:
            for tile in bundle.tiles:
                append_or_create_new_list(tile_to_bundle_map, tile, bundle)
        return tile_to_bundle_map

    def get_bundle_from_directed_tile(self, tile, dir):
        bundle_list = self.tile_to_bundle_map[tile]
        for bundle in bundle_list:
            if bundle.direction == dir:
                return bundle
        # if looped through and none found no bundle exists
        return None

    def cluster_projections_to_bundles(self, cluster_projections):
        all_projections = list(cluster_projections.keys())
        all_projections.sort()
        partitions = []
        part = [all_projections[0]]
        for curr, nxt in zip(all_projections, all_projections[1:]):
            if cluster_projections[curr] == cluster_projections[nxt]:
                part.append(nxt)
            else:
                partitions.append(part)
                part = [nxt]
        partitions.append(part)
        return partitions

    def get_bundles(self):
        bins = dict()
        all_bundles = []
        # initalizes bins to be empty lists
        for direction in DIRECTION_TO_VECTOR:
            bins[direction] = dict()
        # collects lines
        for node in self.legal_orientations:
            if self.legal_orientations[node]:
                for direction in self.legal_orientations[node]:
                    line_key = node[1-np.nonzero(DIRECTION_TO_VECTOR[direction])[0][0]]
                    if line_key not in bins[direction]:
                        bins[direction][line_key] = [node]
                    else:
                        bins[direction][line_key].append(node)
        # sorts bins
        for direction in DIRECTION_TO_VECTOR:
            if len(bins[direction]) > 0:
                direction_idx = 1-np.nonzero(DIRECTION_TO_VECTOR[direction])[0][0]
                for line_cluster in separate_list(list(bins[direction].keys()), 'increasing', sort=True):
                    cluster_projections = dict()
                    for line in line_cluster:
                        for point in bins[direction][line]:
                            projection = point[1-direction_idx]
                            if projection not in cluster_projections:
                                cluster_projections[projection] = set([line])
                            else:
                                cluster_projections[projection].add(line)
                    partitions = self.cluster_projections_to_bundles(cluster_projections)
                    bundles = self.create_subpartitions(cluster_projections, partitions, direction)
                    all_bundles = all_bundles + bundles
        return all_bundles

    def create_subpartitions(self, cluster_projections, partitions, direction):
        bundles = []
        for partition in partitions:
            for lst in separate_list(list(cluster_projections[partition[0]]), 'increasing', sort=True):
                bundle = Bundle(lst, list(cluster_projections.keys()), direction)
                bundles.append(bundle)
        return bundles

    def get_drivable_tiles(self):
        drivable_tiles = []
        non_drivable_nodes = []
        drivable_nodes = []
        for xy in self.nodes:
            if self.grid[xy] == '-':
                symbol = '.'
                non_drivable_nodes.append(xy)
            else:
                drivable_nodes.append(xy)
                symbol = self.grid[xy]
            tile = DrivableTile(xy,symbol)
            drivable_tiles.append(tile)
        return drivable_tiles, drivable_nodes, non_drivable_nodes

    def get_grid(self, csv_filename):
        grid = dict()
        obstacles = dict()
        with open(csv_filename + '.csv', 'rt') as f:
            graph = csv.reader(f)
            for i, row in enumerate(graph):
                for j, item in enumerate(row):
                    if item != 'x':
                        grid[i,j] = item
        return grid

    def get_IO_map(self):
        sources, sinks = self.get_sources_sinks()
        IO_map = IOMap(sources=sources,sinks=sinks,map=dict())
        for source in IO_map.sources:
            IO_map.map[source] = []
            for sink in IO_map.sinks:
                if sink.node in nx.descendants(self.road_map, source.node):
                    IO_map.map[source].append(sink)
        return IO_map

    def get_sources_sinks(self):
        presources = []
        presinks = []
        for tile in self.legal_orientations:
            if self.legal_orientations[tile]:
                for orientation in self.legal_orientations[tile]:
                    inp = tile[0], tile[1], orientation
                    is_sink = self.road_map.out_degree(inp) == 0
                    is_source = self.road_map.in_degree(inp) == 0
                if is_sink:
                    sink = Sink(inp)
                    presinks.append(sink)
                if is_source:
                    source = Source(inp, spawn_probability=self.default_spawn_probability)
                    presources.append(source)
        return presources, presinks

    def get_traffic_lights(self, random_traffic_lights_init):
        def search_along(start, direction, search_length, green_symbol, yellow_symbol, red_symbol):
            nodes = []
            for k in range(search_length):
                node = tuple(np.array(start) + k*np.array(direction))
                if node in self.grid and self.grid[node] == green_symbol:
                    tl_node = TLNode(node, green_symbol, yellow_symbol, red_symbol)
                    nodes.append(tl_node)
            return nodes

        traffic_lights = od()
        for intersection in self.intersections:
            mcorner = intersection.mcorner
            pcorner = intersection.pcorner
            htop = search_along((mcorner[0]-1,mcorner[1]),[0, 1],intersection.width,'⇣','┄','─')
            hbot = search_along((pcorner[0]+1,pcorner[1]),[0, -1],intersection.width,'⇡','┄','─')
            vleft = search_along((mcorner[0],mcorner[1]-1),[1, 0],intersection.height,'⇢','┆','│')
            vright = search_along((pcorner[0],pcorner[1]+1),[-1, 0],intersection.height,'⇠','┆','│')

            htiles = htop + hbot
            vtiles = vleft + vright
            if htiles or vtiles:
                light_id = len(traffic_lights)
                traffic_light = TrafficLight(light_id=light_id,htiles=htiles,vtiles=vtiles, random_init=random_traffic_lights_init)
                traffic_lights[traffic_light] = intersection

        return traffic_lights

    def get_intersection_to_traffic_light_map(self):
        intersection_to_traffic_light_map = dict()
        for traffic_light in self.traffic_lights:
            intersection = self.traffic_lights[traffic_light]
            intersection_to_traffic_light_map[intersection] = traffic_light
        return intersection_to_traffic_light_map

    def get_tile_to_intersection_map(self):
        tile_to_intersection_map = dict()
        for intersection in self.intersections:
            for tile in intersection.tiles:
                tile_to_intersection_map[tile] = intersection
        return tile_to_intersection_map

    def get_intersections(self):
        found_intersections = []
        inspected = []

        def inspect_in(node, direction):
            node = np.array(node)
            direction = np.array(direction)
            k = 0
            check_node = node
            while self.grid[tuple(check_node)] == '+':
                check_node += direction
                k += 1
            return k-1

        for node in self.grid:
            if self.grid[node] == '+' and node not in inspected:
                xp = inspect_in(node, [1, 0])
                yp = inspect_in(node, [0, 1])
                xm = inspect_in(node, [-1, 0])
                ym = inspect_in(node, [0, -1])
                node_p = node[0] + xp, node[1] + yp
                node_m = node[0] - xm, node[1] - ym
                height = node_p[0] - node_m[0] + 1
                width = node_p[1] - node_m[1] + 1
                tiles = []
                for x in range(node[0] - xm, node[0] + xp + 1):
                    for y in range(node[1] - ym, node[1] + yp + 1):
                        inspected.append((x,y))
                        tiles.append((x,y))
                tiles = tuple(tiles)
                new_intersection = Intersection(tiles, node_p, node_m, height, width)
                found_intersections.append(new_intersection)
        return found_intersections

    def get_road_map(self):
        directions = ['south', 'north', 'east', 'west']
        road_map = nx.DiGraph()
        east_neighbors = []
        east_neighbors.append(Neighbor(xyt=(0,1,'east'), weight=1, name='forward'))
        east_neighbors.append(Neighbor(xyt=(-1,1,'east'), weight=2, name='left-lane'))
        east_neighbors.append(Neighbor(xyt=(1,1,'east'), weight=2, name='right-lane'))
        east_neighbors.append(Neighbor(xyt=(-1,1,'north'), weight=2, name='left-turn'))
        east_neighbors.append(Neighbor(xyt=(1,1,'south'), weight=2, name='right-turn'))
        def get_cell_neighbors(cell, direction):
            x, y = cell
            cell_neighbors = []
            angle = Car.convert_orientation(direction)
            rangle = angle / 180.0 * np.pi
            for east_neighbor in east_neighbors:
                xyt = east_neighbor.xyt
                weight = east_neighbor.weight
                neighbor_name = east_neighbor.name
                nex, ney, nangle = xyt
                dx, dy = rotate_vector(np.array([nex, ney]), rangle)
                fangle = Car.convert_orientation((angle + Car.convert_orientation(nangle)) % 360)
                neighbor_xyt = (x+dx, y+dy, fangle)
                cell_neighbors.append(Neighbor(xyt=neighbor_xyt, weight=weight, name=neighbor_name))
            return cell_neighbors

        road_cells = [cell for cell in self.nodes if self.legal_orientations[cell]]

        for cell in road_cells:
            for direction in directions:
                source = (cell[0], cell[1], direction)
                for neighbor in get_cell_neighbors(cell, direction):
                    nex, ney, neheading = neighbor.xyt
                    weight = neighbor.weight
                    neighbor_name = neighbor.name
                    if (nex, ney) in self.legal_orientations and self.legal_orientations[nex, ney]:
                        if neheading in self.legal_orientations[nex, ney]:
                            if neighbor_name in ['left-lane', 'right-lane']:
                                cond1 = len(self.legal_orientations[nex, ney]) == 1
                                cond2 = len(self.legal_orientations[cell]) == 1
                                if cond1 and cond2: # no lane change at intersection
                                    road_map.add_edge(source, neighbor.xyt, weight=weight)
                            elif neighbor_name in ['right-turn']:
                                cond1 = len(self.legal_orientations[nex, ney]) == 1
                                cond2 = len(self.legal_orientations[cell]) == 1
                                if cond1 and cond2:
                                    road_map.add_edge(source, neighbor.xyt, weight=weight)
                            elif neighbor_name in ['left-turn']:
                                road_map.add_edge(source, neighbor.xyt, weight=weight)
                            else:
                                road_map.add_edge(source, neighbor.xyt, weight=weight)
        return road_map

    def get_legal_orientations(self):
        orientation_set = ['←','→','↓','↑','+','⇠','⇢','⇣','⇡']

        def search_all_directions(node):
            found_orientations = []

            def search_in_direction(node, direction):
                curr_node = (node[0] + direction[0], node[1] + direction[1])
                while node in self.grid and self.grid[curr_node] == '+':
                    curr_node = (curr_node[0] + direction[0], curr_node[1] + direction[1])
                return symbol_to_orientation(self.grid[curr_node])

            for direction in DIRECTION_TO_VECTOR:
                dvec = DIRECTION_TO_VECTOR[direction]
                found = search_in_direction(node, dvec)
                if found and found not in found_orientations:
                    found_orientations.append(found)
            return found_orientations

        legal_orientations = dict()
        for node in self.grid:
            symb = self.grid[node]
            if symb not in orientation_set:
                legal_orientations[node] = None
            elif symb != '+':
                legal_orientations[node] = [symbol_to_orientation(symb)]
            else:
                legal_orientations[node] = search_all_directions(node)
        return legal_orientations

class Controller:
    def __init__(self, game):
        self.game = game
    def run_on(self, plant):
        raise NotImplementedError

class RandomController(Controller):
    def __init__(self, plant):
        super(RandomController, self).__init__(plant=plant,game=None)
    def run_on(self, plant):
        plant.apply(random.choice([action for action in
            plant.get_all_ctrl() if plant]))

class ObstacleAvoidingRandomController(Controller):
    def __init__(self, game):
        super(ObstacleAvoidingRandomController, self).__init__(game=game)
    def run_on(self, plant):
        safe_ctrl = []
        for ctrl in plant.get_all_ctrl():
            if all([(occ_state.x, occ_state.y) in self.game.map.drivable_nodes
                for occ_state in plant.query_occupancy(ctrl)]):
                safe_ctrl.append(ctrl)
        if len(safe_ctrl) > 0:
            plant.apply(random.choice(safe_ctrl))
        else:
            RuntimeError('An inevitable collision has occurred!')

class GridScheduler():
    def __init__(self, plant_list, the_map):
        self.request_list = []

class GridTrackingController(Controller):
    def __init__(self, reference):
        self.reference = reference

    def sanity_check(self, plant):
        assert (self.reference[0] == np.array([plant.state.x, plant.state.y])).all()

    def run_on(self, plant):
        self.sanity_check(plant)
        if len(self.reference) > 1:
            next_xy = self.reference[1]
            for ctrl in plant.get_all_ctrl():
                final_state = plant.query_occupancy(ctrl)[-1]
                if ((final_state.x, final_state.y) == next_xy).all():
                    plant.apply(ctrl)
                    self.reference = self.reference[1:]
                    break

class Planner:
    def __init__(self, planning_map):
        self.map = planning_map

    def switch_map(self, new_map):
        self.map = new_map

    def plan(self, start, end):
        raise NotImplementedError

class GridPlanner(Planner):
    def __init__(self, planning_map):
        super(GridPlanner, self).__init__(planning_map=planning_map)
    def get_plan(self, start, end):
        assert start in self.map.drivable_nodes and end in self.map.drivable_nodes
        x0, y0 = start
        x1, y1 = end

        dx = x1 - x0
        x_inc = np.sign(dx)
        dy = y1 - y0
        y_inc = np.sign(dy)

        current = np.array([x0, y0])
        plan = [current]

        x_curr = x0
        y_curr = y0
        for i in range(np.abs(dx)):
            x_curr = x_curr + x_inc
            new_xy = np.array([x_curr, y_curr])
            plan.append(new_xy)
        for j in range(np.abs(dy)):
            y_curr = y_curr + y_inc
            new_xy = np.array([x_curr, y_curr])
            plan.append(new_xy)
        return plan

class Oracle():
    def __init__(self, name):
        self.name = name
        pass
    def evaluate(self, ctrl_action, plant, game):
        raise NotImplementedError

class ReplanProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(ReplanProgressOracle, self).__init__(name='replan_progress')
    def evaluate(self, ctrl_action, plant, game):
        source = (plant.state.x, plant.state.y)
        target = plant.supervisor.current_goal
        current_distance = len(nx.astar_path(game.map.road_map, source, target))
        final_state = plant.query_occupancy(ctrl_action)[-1]
        final_node = final_state.x, final_state.y
        try:
            next_distance = len(nx.astar_path(game.map.road_map, final_node, target))
        except:
            next_distance = np.inf
        return next_distance < current_distance

class PathProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(PathProgressOracle, self).__init__(name='path_progress')
    def evaluate(self, ctrl_action, plant, game):
        def get_dist_to_plan(xy, plan):
            all_dist = [np.sum(np.abs(xy-np.array([p[0], p[1]]))) for p in plan]
            closest_p = np.argmin(all_dist)
            return len(plan)-closest_p-1 + all_dist[closest_p]*100
        current_plan = plant.supervisor.current_plan
        final_state = plant.query_occupancy(ctrl_action)[-1]
        current_xy = np.array([plant.state.x, plant.state.y])
        queried_xy = np.array([final_state.x, final_state.y])
        current_dist = get_dist_to_plan(current_xy, current_plan)
        queried_dist = get_dist_to_plan(queried_xy, current_plan)
        return queried_dist < current_dist

class BundleProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(BundleProgressOracle, self).__init__(name='bundle_progress')
    def evaluate(self, ctrl_action, plant, game):
        def backup_plan_is_ok_from_state(state, current_subgoal):
            tile_sequence_chain = plant.query_backup_plan(state=state)
            # time-stamp, occupancy-tiles, which tile
            last_xy = tile_sequence_chain[-1][-1][-1]
            backup_xy = last_xy
            # heading is unchanged because maximal braking backup plan does not change heading
            backup_dir = state.heading
            try:
                return plant.supervisor.game.map.check_directed_tile_reachability((backup_xy, backup_dir), current_subgoal)
            except:
                return False

        # get current subgoal
        current_subgoal = plant.supervisor.subgoals[0]
        # get bundle corresponding to the current subgoal
        subgoal_bundle = plant.supervisor.game.map.directed_tile_to_bundle(current_subgoal)
        # if subgoal_bundle is None: TODO: delete this?
        #     return False

        current_xy = plant.state.x, plant.state.y
        current_dir = plant.state.heading
        # get bundle corresponding to the current state
        current_bundle = plant.supervisor.game.map.directed_tile_to_bundle((current_xy, current_dir))

        # query next state
        queried_state = plant.query_occupancy(ctrl_action)[-1]
        queried_xy = queried_state.x, queried_state.y
        queried_dir = queried_state.heading

        try:
            # get bundle for the queried state, if bundle doesn't exist, return False
            queried_bundle = plant.supervisor.game.map.directed_tile_to_bundle((queried_xy, queried_dir))
        except:
            return False

        if current_bundle != subgoal_bundle:
            if len(plant.supervisor.subgoals) > 1:
                next_subgoal = plant.supervisor.subgoals[1]
                backup_plan_ok = backup_plan_is_ok_from_state(queried_state, next_subgoal)
                if not backup_plan_ok:
                    return False
            bundle_change_success = (queried_xy, queried_dir) == current_subgoal
            return bundle_change_success
        elif queried_bundle == subgoal_bundle:
            rel_curr = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((current_xy, current_dir))
            rel_next = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((queried_xy, queried_dir))
            rel_goal = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((current_subgoal[0], current_subgoal[1]))

            dlong_curr = rel_goal[1]-rel_curr[1]
            dlong_next = rel_goal[1]-rel_next[1]
            # if dlong_next < 0: TODO: delete?
            #     return False

            dlatt_curr = abs(rel_curr[0]-rel_goal[0])
            dlatt_next = abs(rel_next[0]-rel_goal[0])

            # check if strictly improving longitudinal/lateral distance
            latt_improves = dlatt_next < dlatt_curr
            long_improves = dlong_next < dlong_curr
            improves = latt_improves or long_improves

            # check if at least maintaining longitudinal/lateral distance
            latt_maintains = dlatt_next <= dlatt_curr
            long_maintains = dlong_next <= dlong_curr
            maintains = latt_maintains and long_maintains

            # TODO: separate into two different oracles to distinguish cases
            # only need to check reachability for braking backup plan because this implies reachability for current action
            backup_plan_ok = backup_plan_is_ok_from_state(queried_state, current_subgoal)
            progress_up = improves and maintains and backup_plan_ok
            return progress_up
        else:
            return False

class ImprovementBundleProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(ImprovementBundleProgressOracle, self).__init__(name="improve_progress")
    def evaluate(self, ctrl_action, plant, game):

        # get current state
        current_xy = plant.state.x, plant.state.y
        current_dir = plant.state.heading
        current_directed_tile = (current_xy, current_dir)

        # query next state
        queried_state = plant.query_occupancy(ctrl_action)[-1]
        queried_xy = queried_state.x, queried_state.y
        queried_dir = queried_state.heading
        queried_directed_tile = (queried_xy, queried_dir)
        # get current subgoal
        current_subgoal = plant.supervisor.subgoals[0]
        # if current subgoal is special, namely, attempting to move toward special tile
        # for now, we are assuming default car dynamics; TODO: generalize this
        if current_subgoal in plant.supervisor.game.map.special_goal_tiles:
            return (queried_directed_tile == current_subgoal)
        # if moving away from special tile
        elif current_directed_tile in plant.supervisor.game.map.special_goal_tiles:
            return (queried_directed_tile == current_subgoal)
        # if current subgoal isn't special
        else:
            # get bundle corresponding to the current subgoal
            subgoal_bundle = plant.supervisor.game.map.directed_tile_to_bundle(current_subgoal)
            # get bundle corresponding to the current state
            current_bundle = plant.supervisor.game.map.directed_tile_to_bundle((current_xy, current_dir))

            try:
                # get bundle for the queried state, if bundle doesn't exist, return False
                queried_bundle = plant.supervisor.game.map.directed_tile_to_bundle((queried_xy, queried_dir))
            except:
                return False

            if current_bundle != subgoal_bundle:
                if (queried_xy, queried_dir) == current_subgoal: # bundle change succeeds, assuming default car dynamics
                    return True
                elif (queried_xy, queried_dir) == (current_xy, current_dir): # if doesn't make bundle change
                    return False
                else:
                    return False
            elif queried_bundle == subgoal_bundle:
                rel_curr = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((current_xy, current_dir))
                rel_next = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((queried_xy, queried_dir))
                rel_goal = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((current_subgoal[0], current_subgoal[1]))

                dlong_curr = rel_goal[1]-rel_curr[1]
                dlong_next = rel_goal[1]-rel_next[1]

                dlatt_curr = abs(rel_curr[0]-rel_goal[0])
                dlatt_next = abs(rel_next[0]-rel_goal[0])

                # check if strictly improving longitudinal/lateral distance
                latt_improves = dlatt_next < dlatt_curr
                long_improves = dlong_next < dlong_curr
                improves = latt_improves or long_improves

                # only need to check reachability for braking backup plan because this implies reachability for current action
                return improves
            else:
                return False

class MaintenanceBundleProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(MaintenanceBundleProgressOracle, self).__init__(name='maintain_progress')
    def evaluate(self, ctrl_action, plant, game):
        # get current state
        current_xy = plant.state.x, plant.state.y
        current_dir = plant.state.heading
        current_directed_tile = (current_xy, current_dir)

        # query next state
        queried_state = plant.query_occupancy(ctrl_action)[-1]
        queried_xy = queried_state.x, queried_state.y
        queried_dir = queried_state.heading
        queried_directed_tile = (queried_xy, queried_dir)
        # get current subgoal
        current_subgoal = plant.supervisor.subgoals[0]
        # if current subgoal is special, namely, attempting to move toward special tile
        # for now, we are assuming default car dynamics; TODO: generalize this
        if current_subgoal in plant.supervisor.game.map.special_goal_tiles:
            return (current_directed_tile == queried_directed_tile) or (queried_directed_tile == current_subgoal)
        # if moving away from special tile
        elif current_directed_tile in plant.supervisor.game.map.special_goal_tiles:
            return (current_directed_tile == queried_directed_tile) or (queried_directed_tile == current_subgoal)
        # if current subgoal isn't special
        else:
            # get bundle corresponding to the current subgoal
            subgoal_bundle = plant.supervisor.game.map.directed_tile_to_bundle(current_subgoal)
            # get bundle corresponding to the current state
            current_bundle = plant.supervisor.game.map.directed_tile_to_bundle((current_xy, current_dir))
            try:
                # get bundle for the queried state, if bundle doesn't exist, return False
                queried_bundle = plant.supervisor.game.map.directed_tile_to_bundle((queried_xy, queried_dir))
            except:
                return False

            if current_bundle != subgoal_bundle:
                if (queried_xy, queried_dir) == current_subgoal: # bundle change succeeds, assuming default car dynamics
                    return True
                elif (queried_xy, queried_dir) == (current_xy, current_dir): # if doesn't make bundle change
                    return True
                else:
                    return False
            elif queried_bundle == subgoal_bundle:
                rel_curr = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((current_xy, current_dir))
                rel_next = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((queried_xy, queried_dir))
                rel_goal = plant.supervisor.game.map.directed_tile_to_relative_bundle_tile((current_subgoal[0], current_subgoal[1]))

                dlong_curr = rel_goal[1]-rel_curr[1]
                dlong_next = rel_goal[1]-rel_next[1]

                dlatt_curr = abs(rel_curr[0]-rel_goal[0])
                dlatt_next = abs(rel_next[0]-rel_goal[0])

                # check if at least maintaining longitudinal/lateral distance
                latt_maintains = dlatt_next <= dlatt_curr
                long_maintains = dlong_next <= dlong_curr
                maintains = latt_maintains and long_maintains

                # TODO: separate into two different oracles to distinguish cases
                # only need to check reachability for braking backup plan because this implies reachability for current action
                return maintains
            else:
                return False


class BackUpPlanBundleProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(BackUpPlanBundleProgressOracle, self).__init__(name='backup_plan_progress')
    def evaluate(self, ctrl_action, plant, game):
        def backup_plan_is_ok_from_state(state, current_subgoal):
            tile_sequence_chain = plant.query_backup_plan(state=state)
            # time-stamp, occupancy-tiles, which tile
            last_xy = tile_sequence_chain[-1][-1][-1]
            backup_xy = last_xy
            # heading is unchanged because maximal braking backup plan does not change heading
            backup_dir = state.heading
            try:
                return plant.supervisor.game.map.check_directed_tile_reachability((backup_xy, backup_dir), current_subgoal)
            except:
                return False

        # get current state
        current_xy = plant.state.x, plant.state.y
        current_dir = plant.state.heading
        current_directed_tile = (current_xy, current_dir)

        # query next state
        queried_state = plant.query_occupancy(ctrl_action)[-1]
        queried_xy = queried_state.x, queried_state.y
        queried_dir = queried_state.heading
        queried_directed_tile = (queried_xy, queried_dir)
        # get current subgoal
        current_subgoal = plant.supervisor.subgoals[0]
        # if current subgoal is special, namely, attempting to move toward special tile
        # for now, we are assuming default car dynamics; TODO: generalize this
        if current_subgoal in plant.supervisor.game.map.special_goal_tiles:
            return (current_directed_tile == queried_directed_tile) or (queried_directed_tile == current_subgoal)
        # if moving away from special tile
        elif current_directed_tile in plant.supervisor.game.map.special_goal_tiles:
            return (current_directed_tile == queried_directed_tile) or (queried_directed_tile == current_subgoal)
        # if current subgoal isn't special
        else:
            # get bundle corresponding to the current subgoal
            subgoal_bundle = plant.supervisor.game.map.directed_tile_to_bundle(current_subgoal)
            # get bundle corresponding to the current state
            current_bundle = plant.supervisor.game.map.directed_tile_to_bundle((current_xy, current_dir))
            try:
                # get bundle for the queried state, if bundle doesn't exist, return False
                queried_bundle = plant.supervisor.game.map.directed_tile_to_bundle((queried_xy, queried_dir))
            except:
                return False

            # this is where a bundle change may happen
            if current_bundle != subgoal_bundle:
                if (queried_xy, queried_dir) == current_subgoal: # bundle change succeeds, assuming default car dynamics
                    if len(plant.supervisor.subgoals) > 1:
                        next_subgoal = plant.supervisor.subgoals[1]
                        return backup_plan_is_ok_from_state(queried_state, next_subgoal)
                elif (queried_xy, queried_dir) == (current_xy, current_dir): # if doesn't make bundle change
                    return backup_plan_is_ok_from_state(queried_state, current_subgoal)
                else:
                    return False
            elif queried_bundle == subgoal_bundle:
                # only need to check reachability for braking backup plan because this implies reachability for current action
                return backup_plan_is_ok_from_state(queried_state, current_subgoal)
            else:
                return False


class UnprotectedLeftTurnOracle(Oracle):
    def __init__(self):
        super(UnprotectedLeftTurnOracle, self).__init__(name='unprotected_left_turn')
    def evaluate(self, ctrl_action, plant, game):
        current_directed_tile = (plant.state.x, plant.state.y), plant.state.heading
        queried_state = plant.query_occupancy(ctrl_action)[-1]
        queried_xy = queried_state.x, queried_state.y
        queried_dir = queried_state.heading
        queried_directed_tile = (queried_xy, queried_dir)
        # check if agent needs to reason about making an unprotected left turn
        if current_directed_tile in game.map.all_left_turns:
            current_bundle = game.map.directed_tile_to_bundle(current_directed_tile)
            if queried_directed_tile not in game.map.left_turn_tiles[current_bundle][current_directed_tile]: # if not turning left
                return True
            else: # if commit to turning
                opposing_bundle, relative_occupancy = game.map.left_turn_to_opposing_traffic_bundles[current_directed_tile]
                relative_tiles = [(idx, relative_occupancy[idx]) for idx in range(len(relative_occupancy))]
                relative_tiles.reverse() # sort so temporal order is achieved
                gaps = []
                fake_heading = Car.convert_orientation(Car.convert_orientation(plant.state.heading) + 180)
                for occupancy_tile in relative_tiles:
                    abs_x, abs_y = opposing_bundle.relative_coordinates_to_tile(occupancy_tile)
                    fake_state = Car.hack_state(plant.state, x=abs_x, y=abs_y, heading=fake_heading)
                    lead_agent = plant.find_lead_agent(fake_state, same_heading_required=False)
                    if lead_agent is None:
                        pass
                    else:
                        gap = max(abs_x-lead_agent.state.x, abs_y-lead_agent.state.y)
                        current_intersection = game.map.tile_to_intersection_map[current_directed_tile[0]]
                        # get traffic light
                        traffic_light = game.map.intersection_to_traffic_light_map[current_intersection]
                        # TODO: complete gap conditions
                        if gap > 6:
                            pass
                        else:
                            return False
                return True
        else: # if the agent is not trying to perform a left turn
            return True

class TrafficLightOracle(Oracle):
    def __init__(self):
        super(TrafficLightOracle, self).__init__(name='traffic_light')

    def tile_sequence_not_running_a_red_light_on_N_turn(self, tile_sequence, game, N):
        light_checks = []
        for tile in tile_sequence:
            if tile in game.map.tile_to_traffic_light_map:
                # figure out traffic_light orientation
                legal_orientation = game.map.legal_orientations[tile][0]
                light_checks.append([tile, legal_orientation])

        for light_check in light_checks:
            light_tile, legal_orientation = light_check
            will_be_crossing = self.check_if_crossing(light_tile, tile_sequence, legal_orientation)
            # get traffic light
            traffic_light = game.map.tile_to_traffic_light_map[light_tile]
            red_light_on = self.check_if_light_red_in_N_turns(traffic_light, legal_orientation, N)
            if will_be_crossing and red_light_on:
                return False
        return True

    def action_not_running_a_red_light(self, ctrl_action, plant, game):
        occ_states = plant.query_occupancy(ctrl_action)
        occ_tiles = [(state.x, state.y) for state in occ_states]
        return self.tile_sequence_not_running_a_red_light_on_N_turn(occ_tiles, game, N=1)

    def check_if_light_red_in_N_turns(self, traffic_light, direction, N):
        future_lights = traffic_light.check_light_N_turns_from_now(N)
        if direction in ['east', 'west']:
            color, _ = future_lights['vertical']
        elif direction in ['north', 'south']:
            color, _ = future_lights['horizontal']
        return color == 'red'

    def check_if_crossing(self, light_tile, tiles, direction):
        if direction in ['west', 'east']:
            k = 1
        elif direction in ['north', 'south']:
            k = 0
        light_tile_projection = light_tile[k]
        tile_projections = [tile[k] for tile in tiles]
        if direction in ['east', 'south']:
            return np.min(tile_projections) <= light_tile_projection and np.max(tile_projections) > light_tile_projection
        elif direction in ['west', 'north']:
            return np.min(tile_projections) < light_tile_projection and np.max(tile_projections) >= light_tile_projection

    def backup_plant_will_still_be_ok(self, ctrl_action, plant, game):
        next_state = plant.query_occupancy(ctrl_action)[-1]
        tile_sequence_chain = plant.query_backup_plan(state=next_state)
        for tile_sequence in tile_sequence_chain:
            N, required_tiles = tile_sequence
            if self.tile_sequence_not_running_a_red_light_on_N_turn(required_tiles, game, N=N):
                pass
            else:
                return False
        return True

    def evaluate(self, ctrl_action, plant, game):
        action_not_running_a_red_light = self.action_not_running_a_red_light(ctrl_action, plant, game)
        backup_plant_will_still_be_ok = self.backup_plant_will_still_be_ok(ctrl_action, plant, game)
        # if you're at the critical right-turn tile and red light
        bundle = game.map.directed_tile_to_bundle(((plant.state.x, plant.state.y), plant.state.heading))
        # special left turn lane bandage
        if bundle is None: 
            #print(plant.state.x, plant.state.y, plant.state.heading)
            return False
        # check whether the right-turn action is okay
        if ((plant.state.x, plant.state.y), plant.state.heading) in game.map.right_turn_tiles[bundle] and ctrl_action['steer'] == 'right-turn':
            traffic_light = game.map.tile_to_traffic_light_map[(plant.state.x, plant.state.y)]
            light_is_red = self.check_if_light_red_in_N_turns(traffic_light, plant.state.heading, 0) # N=0
            if light_is_red: 
                # check if right turn is valid
                return plant.check_right_turn_is_clear(ctrl_action)
            else:
                return action_not_running_a_red_light and backup_plant_will_still_be_ok
        else: 
            return action_not_running_a_red_light and backup_plant_will_still_be_ok

# oracle in charge of checking actions of agents if they are in an intersection
# action is invalid if agent is in intersection and wants to change lanes
class TrafficIntersectionOracle(Oracle):
    def __init__(self):
        super(TrafficIntersectionOracle, self).__init__(name='traffic_intersection')
    def evaluate(self, ctrl_action, plant, game):
        def check_heading_matches_bundle_heading(bundles, heading):
            for bundle in bundles:
                if bundle.direction == heading: 
                    return True
            return False

        # if agent isn't in intersection return true
        if game.map.legal_orientations[(plant.state.x, plant.state.y)] is None:
            return True
        elif len(game.map.legal_orientations[(plant.state.x, plant.state.y)]) <= 1: 
            return True
        # else check if action is a lane change move (which isn't allowed)
        else: 
            ego_tile = plant.state.x, plant.state.y
            ego_heading = plant.state.heading
            bundles = game.map.tile_to_bundle_map[plant.state.x,plant.state.y]
            # check if ego_heading aligns with any of bundles at the tile
            chk_heading_match = check_heading_matches_bundle_heading(bundles, ego_heading)

            # TODO: change 
            if not chk_heading_match or (ego_tile, ego_heading) in game.map.special_goal_tiles:
                #st()
                return True
            # else don't make a lane change in an intersection
            else: 
                if ctrl_action['steer'] == 'left-lane' or ctrl_action['steer'] == 'right-lane':
                    return False
                else:
                    return True

class TrafficLightTurningLanesOracle(Oracle):
    def __init__(self):
        super(TrafficLightTurningLanesOracle, self).__init__(name='traffic_light_turning_lanes')
    def evaluate(self, ctrl_action, plant, game):
        pass 


#TODO: improve some calc here...
class BackupPlanSafetyOracle(Oracle):
    def __init__(self):
        super(BackupPlanSafetyOracle, self).__init__(name='backup_plan_safety')
    def evaluate(self, ctrl_action, plant, game):
        # check if collision occurs by taking that action
        collision_chk = plant.check_collision_in_bubble(ctrl_action)
        if collision_chk: 
            return False
        else: 
            # check if makes collision with agent during action
            next_state = plant.query_occupancy(ctrl_action)[-1]

            x, y, heading, v = next_state.x, next_state.y, next_state.heading, next_state.v
            lead_agent = plant.find_lead_agent(state=next_state)

            if lead_agent:
                #plant.lead_agent = lead_agent.state.__tuple__()
                x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
                gap_curr = ((x_a-x)**2 + (y_a-y)**2)**0.5
                return plant.compute_gap_req(lead_agent.a_min, v_a, plant.a_min, v) <= gap_curr
            else:
                return True

class StaticObstacleOracle(Oracle):
    def __init__(self):
        super(StaticObstacleOracle, self).__init__(name='static_obstacle')
    def evaluate(self, ctrl_action, plant, game):
        # check if action is safe
        next_occupancy = plant.query_occupancy(ctrl_action)
        action_is_safe = all([(occ_state.x, occ_state.y) in
            game.map.drivable_nodes for occ_state in
            next_occupancy])
        if not action_is_safe:
            return False
        else:
        # check if backup plan would be safe
            next_state = next_occupancy[-1]
            tile_sequence_chain = plant.query_backup_plan(state=next_state)
            tiles = list(od.fromkeys([tile for tiles in [tile_turn[-1]
                for tile_turn in tile_sequence_chain] for tile in
                tiles]))
            for tile in tiles:
                if not (tile in game.map.drivable_nodes):
                    return False
            return True

    def backup_plant_will_still_be_ok(self, ctrl_action, plant, game):
        next_state = plant.query_occupancy(ctrl_action)[-1]
        for tile_sequence in tile_sequence_chain:
            N, required_tiles = tile_sequence
        return True

class LegalOrientationOracle(Oracle):
    """
    check if next heading is legal

    """
    def __init__(self):
        super(LegalOrientationOracle, self).__init__(name='legal_orientation')
    def evaluate(self, ctrl_action, plant, game):
        final_state = plant.query_occupancy(ctrl_action)[-1]
        final_node = final_state.x, final_state.y
        if final_node in game.map.drivable_nodes:
            legal_headings = game.map.legal_orientations[final_node]
            if not legal_headings: # check if node is off road
                return False
            else: # check if next heading is legal
                return final_state.heading in legal_headings
        else:
            return False # if node is an obstacle or out of bounds

class OracleController(Controller):
    def __init__(self, game, oracle_set):
        super(OracleController, self).__init__(game=game)
        self.oracle_set = oracle_set
    def run_on(self, plant):
        sat_ctrl = []
        for ctrl in plant.get_all_ctrl():
            if all([oracle.evaluate(ctrl, plant, self.game) for oracle in self.oracle_set]):
                sat_ctrl.append(ctrl)
        if len(sat_ctrl) > 0:
            plant.apply(random.choice(sat_ctrl))
        else:
            RuntimeError('No satisfactory control input found for these oracles!')

class SpecificationStructureController(Controller):
    def __init__(self, game, specification_structure):
        super(SpecificationStructureController, self).__init__(game=game)
        self.specification_structure = specification_structure

    def run_on(self, plant):
        '''def ctrl_dict_to_tuple(ctrl):
            return (ctrl['steer'], ctrl['acceleration'])

        all_ctrls = plant.get_all_ctrl()
        spec_struct_trace = {}

        for ctrl in all_ctrls:
            score = 0
            scores = {}
            for oracle in self.specification_structure.oracle_set:
                o_score = oracle.evaluate(ctrl, plant, self.game)
                o_tier = self.specification_structure.tier[oracle]
                try:
                    score += int(o_score) * self.specification_structure.tier_weights[o_tier]
                except:
                    pass
                scores[oracle.name] = o_score
            # save data
            scores['total'] = score
            spec_struct_trace[ctrl_dict_to_tuple(ctrl)] = scores

        # save data as agent attribute
        plant.spec_struct_trace = spec_struct_trace'''

        # choose action according to action selection strategy
        ctrl = plant.action_selection_strategy()
        plant.apply(ctrl)


class SupervisoryController():
    def _init__(self):
        self.plant = None
        self.game = None

    def add_headings_to_goals(self, goals):
        new_goals = []
        for goal in goals:
            if len(goal) < 3:
                heading = self.game.map.legal_orientations[goal][0] # take first one if multiple
                new_goals.append((goal[0], goal[1], heading))
            else:
                new_goals.append(goal)
        return new_goals

    def set_plant(self, plant):
        plant.supervisor = self
        self.plant = plant
        self.current_goal, self.current_plan = self.get_next_goal_and_plan()

    def get_next_goal_and_plan(self):
        raise NotImplementedError\

    def check_goals(self):
        raise NotImplementedError\

    def run(self):
        self.check_goals()

class GoalExit(SupervisoryController):
    def __init__(self, game, goals=None):
        super(SupervisoryController, self).__init__()
        self.game = game
        self.goals = goals[0] # only consider first goal
    def get_next_goal_and_plan(self):
        next_goal = self.goals
        source = (self.plant.state.x, self.plant.state.y, self.plant.state.heading)
        target = next_goal
        next_plan = nx.astar_path(self.game.map.road_map, source, target)
        return next_goal, next_plan
    def check_goals(self):
        if self.plant:
            if np.sum(np.abs(np.array([self.plant.state.x, self.plant.state.y]) - np.array([self.current_goal[0], self.current_goal[1]]))) <= 1: # if close enough
                self.game.agent_set.remove(self.plant)

class GoalCycler(SupervisoryController):
    def __init__(self, game, goals=None):
        super(SupervisoryController, self).__init__()
        self.game = game
        self.goals = cycle(self.add_headings_to_goals(goals))

    def get_next_goal_and_plan(self):
        next_goal = next(self.goals)
        source = (self.plant.state.x, self.plant.state.y, self.plant.state.heading)
        target = next_goal
        next_plan = nx.astar_path(self.game.map.road_map, source, target)
        return next_goal, next_plan

    def check_goals(self):
        if self.plant:
            if np.sum(np.abs(np.array([self.plant.state.x, self.plant.state.y]) - np.array([self.current_goal[0], self.current_goal[1]]))) == 0: # if close enough
                self.current_goal, self.current_plan = self.get_next_goal_and_plan()

class BundleGoalExit(SupervisoryController):
    def __init__(self, game, goals=None):
        super(SupervisoryController, self).__init__()
        self.game = game
        self.goals = goals[0] # only consider first goal
        self.subgoals = None

    def get_next_goal_and_plan(self):
        next_goal = self.goals
        source = ((self.plant.state.x, self.plant.state.y), self.plant.state.heading)
        target = ((next_goal[0], next_goal[1]), next_goal[2])
        next_plan = self.game.map.get_bundle_plan(source, target)
        self.subgoals = self.get_subgoals(next_plan)
        return next_goal, next_plan

    def get_subgoals(self, plan):
        subgoals = []
        for node in plan:
            if isinstance(node[1], str):
                subgoals.append(node)
            else:
                for subnode in node:
                    subgoals.append(subnode)
        subgoals = subgoals[1:] # remove first node
        return subgoals


    def check_subgoals(self):
        if self.plant:
            if np.sum(np.abs(np.array([self.plant.state.x, self.plant.state.y]) - np.array([self.subgoals[0][0][0], self.subgoals[0][0][1]]))) == 0:
                self.subgoals = self.subgoals[1:]

    def check_goals(self):
        self.check_subgoals()
        if self.plant:
            if np.sum(np.abs(np.array([self.plant.state.x, self.plant.state.y]) - np.array([self.current_goal[0], self.current_goal[1]]))) == 0: # if close enough
                self.game.agent_set.remove(self.plant)

class SpecificationStructure():
    def __init__(self, oracle_list, oracle_tier):
        assert len(oracle_list) == len(oracle_tier), 'list of oracles must have the same length as corresponding list of tiers'
        self.oracle_set = oracle_list
        self.tier = self.set_tier(oracle_tier)
        self.tier_weights = self.set_tier_weights(oracle_tier)

    def set_tier(self, oracle_tier):
        tier = dict()
        for idx, oracle in enumerate(self.oracle_set):
            tier[oracle] = oracle_tier[idx]
        return tier

    def set_tier_weights(self, oracle_tier):
        def num(tier):
            return np.sum(np.array(oracle_tier) == tier)
        all_tiers = np.sort(list(set(oracle_tier)))[::-1]
        tier_weights = dict()
        tier_weights[all_tiers[0]] = 1
        for idx in range(1, len(all_tiers)):
            tier = all_tiers[idx]
            next_lower_tier = all_tiers[idx-1]
            tier_weights[tier] = (num(next_lower_tier)+1)*tier_weights[next_lower_tier]
        return tier_weights

class TrafficLight:
    # 15, 4, 4
    def __init__(self, light_id, htiles, vtiles, t_green=20,t_yellow=3,t_buffer=10, random_init=True):
        self.id = light_id
        self.durations = od()
        self.durations['green'] = t_green
        self.durations['yellow'] = t_yellow
        self.t_buffer = t_buffer
        self.durations['red'] = self.durations['green'] + self.durations['yellow'] + self.t_buffer * 2
        self.states = cycle([color for color in self.durations])
        if random_init:
            self.hstate = np.random.choice([color for color in self.durations])
            self.htimer = np.random.choice(self.durations[self.hstate])
        else:
            self.htimer = 0
            self.hstate = next(self.states)
        self.htiles = htiles
        self.vtiles = vtiles

    def get_id(self):
        return id(self)

    def set_traffic_lights(self, hstate, htimer):
        #traffic_light = self.tile_to_traffic_light_map[traffic_tile]
        assert hstate in ['red', 'yellow', 'green']
        assert htimer < self.durations[hstate]
        self.hstate = hstate
        self.htimer = htimer

    def check_light_N_turns_from_now(self, N):
        dummy_traffic_light = cp.deepcopy(self)
        for t in range(N):
            dummy_traffic_light.run()
        return {'horizontal': dummy_traffic_light.get_hstate(), 'vertical': dummy_traffic_light.get_vstate()}

    def run(self):
        self.htimer += 1
        if self.htimer >= self.durations[self.hstate]:
            self.hstate = next(self.states)
            self.htimer = 0

    def get_hstate(self):
        return self.hstate, self.htimer

    def get_vstate(self):
        if self.hstate == 'green' or self.hstate == 'yellow':
            color = 'red'
            if self.hstate == 'green':
                timer = self.t_buffer + self.htimer
            elif self.hstate == 'yellow':
                timer = self.t_buffer + self.durations['green'] + self.htimer
        else: # if red
            if self.htimer < self.t_buffer:
                color = 'red'
                timer = self.htimer + self.durations['green'] + self.durations['yellow'] + self.t_buffer
            elif self.htimer < self.t_buffer + self.durations['green']:
                color = 'green'
                timer = self.htimer - self.t_buffer
            elif self.htimer < self.t_buffer + self.durations['green'] + self.durations['yellow']:
                color = 'yellow'
                timer = self.htimer - self.t_buffer - self.durations['green']
            else:
                color = 'red'
                timer = self.htimer - self.t_buffer - self.durations['green'] - self.durations['yellow']
        return color, timer

    def get_drawables(self):
        def state_to_symb(state, tile):
            if state == 'green':
                return tile.green
            elif state == 'yellow':
                return tile.yellow
            elif state == 'red':
                return tile.red
        drawables = []
        for tile in self.htiles:
            drawable = Drawable(tile.xy, state_to_symb(self.get_hstate()[0], tile))
            drawables.append(drawable)
        for tile in self.vtiles:
            drawable = Drawable(tile.xy, state_to_symb(self.get_vstate()[0], tile))
            drawables.append(drawable)
        return drawables

    def get_hstate_rem(self):
        color, timer = self.get_hstate()
        return color, self.durations[color] - timer

    def get_vstate_rem(self):
        color, timer = self.get_vstate()
        return color, self.durations[color] - timer

class Source:
    def __init__(self, node, spawn_probability):
        self.node = node
        self.p = spawn_probability

class Sink:
    def __init__(self, node):
        self.node = node

def get_default_car_ss():
    backup_plan_safety_oracle = BackupPlanSafetyOracle()
    static_obstacle_oracle = StaticObstacleOracle()
    traffic_light_oracle = TrafficLightOracle()
    legal_orientation_oracle = LegalOrientationOracle()
    backup_plan_progress_oracle = BackUpPlanBundleProgressOracle()
    maintenance_progress_oracle = MaintenanceBundleProgressOracle()
    improvement_progress_oracle = ImprovementBundleProgressOracle()
    traffic_intersection_oracle = TrafficIntersectionOracle()
    unprotected_left_turn_oracle = UnprotectedLeftTurnOracle()
    oracle_set = [static_obstacle_oracle, traffic_light_oracle,
            legal_orientation_oracle, backup_plan_progress_oracle,
            maintenance_progress_oracle, improvement_progress_oracle,
            backup_plan_safety_oracle, unprotected_left_turn_oracle, 
            traffic_intersection_oracle] # type: List[Oracle]
    specification_structure = SpecificationStructure(oracle_set, [1, 2, 2, 3, 4, 4, 1, 1, 2])
    return specification_structure

def create_default_car(source, sink, game):
    ss = get_default_car_ss()
    spec_struct_controller = SpecificationStructureController(game=game,specification_structure=ss)
    start = source.node
    end = sink.node
    car = Car(x=start[0],y=start[1],heading=start[2],v=0,v_min=0,v_max=3, a_min=-1,a_max=1)
    car.set_controller(spec_struct_controller)
    supervisor = BundleGoalExit(game=game, goals=[end])
    car.set_supervisor(supervisor)
    return car

def run(runnable_set):
    for runnable in runnable_set:
        runnable.run()

class QuasiSimultaneousGame(Game):
    def __init__(self, game_map):
        super(QuasiSimultaneousGame, self).__init__(game_map=game_map)
        self.bundle_to_agent_precedence = self.get_bundle_to_agent_precedence()

    def get_bundle_to_agent_precedence(self):
        bundle_to_agent_precedence = dict()
        for bundle in self.map.bundles:
            bundle_to_agent_precedence[bundle] = dict()
        for agent in self.agent_set:
            x, y, heading = agent.state.x, agent.state.y, agent.state.heading
            # if in special tile, rotate by another 90 degrees; TODO: remove this bandage
            if ((x, y), heading) in self.map.special_goal_tiles:
                heading = Car.convert_orientation(Car.convert_orientation(heading) + 90)
            try:
                bundles = self.map.tile_to_bundle_map[(x,y)]
            except:
                bundles = []
            for bundle in bundles:
                if bundle.direction == heading:
                    longitudinal_precedence = bundle.tile_to_relative_length((x,y))
                    append_or_create_new_list(bundle_to_agent_precedence[bundle], \
                                              longitudinal_precedence, \
                                              agent)
        return bundle_to_agent_precedence

    # assuming ego and all agents in agent_set belong to the same bundle
    def get_agents_with_higher_precedence(self, ego, agent_set):
        higher_pred = []
        ego_tile = ego.state.x, ego.state.y
        ego_heading = ego.state.heading
        bundle = self.map.directed_tile_to_bundle((ego_tile, ego_heading))
        ego_score = bundle.tile_to_relative_length(ego_tile)
        for agent in agent_set:
            agent_tile = agent.state.x, agent.state.y
            agent_score = bundle.tile_to_relative_length(agent_tile)
            if agent_score >= ego_score:
                higher_pred.append(agent)
        return higher_pred

    def set_agent_intentions(self):
        for agent in self.agent_set:
            agent.set_intention()

    def resolve_precedence(self):
        self.bundle_to_agent_precedence = self.get_bundle_to_agent_precedence()

    def sys_step(self):
        # set all agent intentions
        self.set_agent_intentions()
        # call resolve_conflicts
        self.send_and_receive_conflict_requests()
        # resolve precedence
        self.resolve_precedence()
        active_agents = []
        for bundle in self.map.bundles:
            precedence_list = list(self.bundle_to_agent_precedence[bundle].keys())
            precedence_list.sort(reverse=True)
            for precedence in precedence_list:
                for agent in self.bundle_to_agent_precedence[bundle][precedence]:
                    agent.run()
                    active_agents.append(agent)

    def play_step(self):
        #for agent in self.agent_set:
            #print(agent.state)
            #print(agent.supervisor.goals)
        self.sys_step()
        self.env_step()

def play_fixed_agent_game_karena_debug(num_agents, game):
    # manually defining some cars to test
    def make_car(game, x=8, y=2, heading='east', v=0):
        ss = get_default_car_ss()
        spec_struct_controller = SpecificationStructureController(game=game,specification_structure=ss)
        end = (x, y+25, heading)
        v_min = 0
        v_max = 3
        #v = random.choice(np.arange(2, v_max+1))
        car = Car(x=x,y=y,heading=heading,v=v,v_min=v_min,v_max=v_max, a_min=-2,a_max=2)
        car.set_controller(spec_struct_controller)
        supervisor = BundleGoalExit(game=game, goals=[end])
        car.set_supervisor(supervisor)
        return car

    # add all agents to the game
    for i in range(num_agents):
        game.add_agent(make_car(game))
    print("starting")
    game.play(outfile=output_filename, t_end=1)

# restart scenario from saved traces at given t_index
def start_game_from_trace(filename, t_index):
    # make a car from trace
    # car_trace = (x, y, heading, v, agent_color, bubble, goals, param)
    def make_car(game, car_trace):
        x, y, heading, v, agent_color, bubble, goal, param = car_trace
        ss = get_default_car_ss()
        spec_struct_controller = SpecificationStructureController(game=game,specification_structure=ss)
        end = goal
        car = Car(x=x,y=y,heading=heading,v=v,v_min=param['v_min'],v_max=param['v_max'], a_min=param['a_min'],a_max=param['a_max'])
        car.set_controller(spec_struct_controller)
        supervisor = BundleGoalExit(game=game, goals=[end])
        car.set_supervisor(supervisor)
        return car

    # traffic light trace (x, y, state, traffic_light.htimer, traffic_light.durations)
    def get_traffic_light_from_xy(the_map, xy):
        # if traffic light has tile with xy position
        for traffic_light in the_map.traffic_lights:
            for tile in traffic_light.htiles:
                if xy == tile.xy: return traffic_light
            for tile in traffic_light.vtiles:
                if xy == tile.xy: return traffic_light
        return None

    # load the file
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)

    traffic_light_traces = traces[t_index]['lights']
    agent_traces = traces[t_index]['agents']

    map_name = traces['map_name']
    spawn_prob = traces['spawn_probability']
    the_map = Map(map_name, default_spawn_probability=0)
    game = QuasiSimultaneousGame(game_map=the_map)

    # reset all the traffic lights according to traces
    traffic_light_reset_dict = {traffic_light.get_id(): False for traffic_light in the_map.traffic_lights}
    for traffic_light_node in traffic_light_traces:
        # orientation is horizontal or vertical
        x, y, state, htimer, orientation, durations = traffic_light_node
        if orientation == 'horizontal':
            traffic_light = get_traffic_light_from_xy(the_map, (x,y))

            # set traffic lights that haven't been set yet
            if not traffic_light_reset_dict[traffic_light.get_id()]:
                traffic_light.set_traffic_light(state, htimer)
                traffic_light_reset_dict[traffic_light.get_id()] = True

    # intialize all the cars according to traces
    for agent_trace in agent_traces:
        ag = make_car(game, agent_trace)
        game.add_agent(ag)

    # start the game
    game.play(t_end=100)

def print_debug_info(filename):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    # print collision dictionary entries
    print("Collisions")
    for key, value in traces['collision_dict'].items():
        print(key, value)

    # print out of bounds dictionary entries
    print("Out of bounds")
    for key, value in traces['out_of_bounds_dict'].items():
        print(key, value)

    #print(traces['unsafe_joint_state_dict'])
    

if __name__ == '__main__':
    seed = 15
    np.random.seed(seed)
    random.seed(seed)
    the_map = Map('./maps/city_blocks_small', default_spawn_probability=0.3)
    output_filename = 'game.p'

    # play a normal game
    game = QuasiSimultaneousGame(game_map=the_map)
    game.play(outfile=output_filename, t_end=100)
    #game.animate(frequency=0.01)

    # print debug info 
    debug_filename = os.getcwd()+'/saved_traces/game.p'
    print_debug_info(debug_filename)

    # play debugged game 

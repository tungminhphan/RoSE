'''

    File name: rose.py
    Authors: Tung Phan, Karena Cai
    Date created: 1/10/2020
'''
import itertools
from typing import List, Any
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
import collections
import os
import _pickle as pickle
import json
import copy as cp
from c_tools import *
#from plotting import traces_to_animation

DEBUGGING = False
FIND_LEAD_AGENT_THRESHOLD = 15 #TODO: compute this automatically
CAR_COLORS = ['blue', 'brown', 'gray', 'green', 'light_green', 'orange', 'red', 'yellow']
IOMap = namedtuple('IOMap', ['sources','sinks','map'])
TLNode = namedtuple('TLNode', ['xy', 'green', 'yellow', 'red'])
Intersection = namedtuple('Intersection', ['tiles', 'pcorner', 'mcorner', 'height', 'width'])
Neighbor = namedtuple('Neighbor', ['xyt', 'weight', 'name'])
DIRECTION_TO_VECTOR  = od()
DIRECTION_TO_VECTOR['east'] = [0, 1]
DIRECTION_TO_VECTOR['west'] = [0, -1]
DIRECTION_TO_VECTOR['north'] = [-1, 0]
DIRECTION_TO_VECTOR['south'] = [1, 0]
AGENT_CHAR = [str(i) for i in range(10)]
CHOSEN_IDs = []

def set_seed(seed, other_param=0):
    #other_param = 0
    if seed is not None:
        np.random.seed(seed+other_param)
        random.seed(seed+other_param)
        np.random.RandomState(seed+other_param)

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
            setattr(self, state_var, kwargs.get(state_var))

    def __str__(self):
        printed = []
        for state_var in self.state_variable_names:
            printed.append(state_var + ' = ' + str(getattr(self, state_var)))
        return self.agent_name + 'State(' + ', '.join(printed) + ')'

    def __tuple__(self):
        tup = ()
        for i, state_var in enumerate(self.state_variable_names):
            a = getattr(self, state_var)
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
        return self.id

    @classmethod
    def hack_state(cls, state, **kwargs):
        new_state = AgentState(agent_name=state.agent_name, state_variable_names=state.state_variable_names)
        for name in new_state.state_variable_names:
            # just copy from state
            is_not_none = kwargs.get(name) != None
            setattr(new_state, name, getattr(state, name))
            if is_not_none:
                # update from is_not_none input
                setattr(new_state, name, kwargs.get(name))
        return new_state

    def __init__(self, **kwargs):

        if 'agent_name' in kwargs:
            self.agent_name = kwargs.get('agent_name')
        else:
            self.agent_name = 'Agent'
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

    def get_drawables(self):
        drawables = []
        drawable = Drawable((self.state.x, self.state.y), self.get_symbol())
        drawables.append(drawable)
        return drawables

    # return a random color from an array
    def get_random_color(self):
        set_seed(self.seed, self.car_count)
        color = np.random.choice(CAR_COLORS)
        #print('car color')
        #print(color)
        return color

    def get_symbol(self):
        if not self.symbol:
            return 'o'
        else:
            return self.symbol

    def run(self):
        assert self.controller, "Agent has no controller"
        self.controller.run_on(self)
        if self.supervisor:
            self.supervisor.run()

    def set_attributes(self, kwargs_dict):
        for attr in self.attributes:
            setattr(self, attr, kwargs_dict.get(attr))

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
        #print(ctrl)
        self.state = self.query_next_state(ctrl)
        self.supervisor.game.update_occupancy_dict_for_one_agent(self, self.prior_state)

        # save action that was chosen
        self.ctrl_chosen = ctrl
        self.check_out_of_bounds(self, self.prior_state, ctrl, self.state)
        
        # check whether the updated joint state is safe
        if DEBUGGING:
            chk_joint_safety = self.check_joint_state_safety(return_list=True)
            self.supervisor.game.unsafe_joint_state_dict[self.supervisor.game.time] = self.check_joint_state_safety(return_list=True)
        else:
            chk_joint_safety = True
            self.supervisor.game.unsafe_joint_state_dict[self.supervisor.game.time] = True

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
            next_state = Agent.hack_state(state, y = state.y - 1)
        elif ctrl == 'down':
            next_state = Agent.hack_state(state, y = state.y + 1)
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
        attributes = ['v_min', 'v_max', 'a_min', 'a_max', 'car_count', 'seed']
        state_variable_names = ['x', 'y', 'heading', 'v']
        #self.seed = 1111
        super(Car, self).__init__(attributes=attributes, agent_name=agent_name, state_variable_names=state_variable_names, **kwargs)
        self.acc_vals = np.arange(self.a_max, self.a_min-1,-1)
        self.default_state = Car.hack_state(self.state, x=0, y=0, heading='east', v=0)
        self.default_bubbles = self.get_default_bubbles()
        self.default_filtered_bubbles = self.get_default_bubbles(front_back_only=True)
        self.bubble = None # this should update at every time step
        self.agents_in_bubble = [] # this should update at every time step
        self.send_conflict_requests_to = [] # list of agents
        self.received_conflict_requests_from = [] # list of agents
        self.agent_max_braking_not_enough = None
        self.token_count = 0
        self.intention = None
        self.turn_signal = None
        self.is_winner = None
        self.id = self.set_id()
        if 'agent_color' in kwargs:
            self.agent_color = kwargs.get('agent_color')
        else:
            self.agent_color = self.get_random_color()

        # attributes for saving agent info
        self.spec_struct_trace = od()
        self.ctrl_chosen = None
        self.unsafe_joint_state_dict = od()
        self.straight_action_eval = od()
        self.action_selection_flags = ()
        self.token_count_before = 0
        self.left_turn_gap_arr = []
        self.clearance_straight_info = None

        self.conflict_winner_sv = None
        self.received_sv = []
        self.sent_sv = []
        self.token_count_sv = 0
        self.agent_max_braking_not_enough_sv = None
        self.agents_in_bubble_sv = []
        self.agents_checked_for_conflict_sv = []
        self.agents_in_bubble_before_sv = []

        #self.lead_vehicle = None
        self.lead_agent = None
        self.prior_state = None

    def apply(self, ctrl):
        self.prior_state = self.state
        #print(self.prior_state)
        # check for collision with any of the other agents
        self.check_collision(ctrl)
        self.state = self.query_next_state(ctrl)
        self.supervisor.game.update_occupancy_dict_for_one_agent(self, self.prior_state)
        # save action that was chosen
        self.ctrl_chosen = ctrl
        self.check_out_of_bounds(self, self.prior_state, ctrl, self.state)
        # check whether the updated joint state is safe
        if DEBUGGING:
            chk_joint_safety = self.check_joint_state_safety(return_list=True)
            self.supervisor.game.unsafe_joint_state_dict[self.supervisor.game.time] = self.check_joint_state_safety(return_list=True)
        else:
            chk_joint_safety = True
            self.supervisor.game.unsafe_joint_state_dict[self.supervisor.game.time] = True

    # for reproducibility
    def set_id(self):
        set_seed(self.seed, self.car_count)
        while True:
            random_id = np.random.uniform()
            if random_id not in CHOSEN_IDs:
                CHOSEN_IDs.append(random_id)
                #print("car id")
                #print(random_id)
                return random_id

    @classmethod
    def get_ctrl_od(cls, acc, steer):
        ctrl = od()
        ctrl['acceleration'] = acc
        ctrl['steer'] = steer
        return ctrl

    # optimized hack_state method for Car agents
    @classmethod
    def hack_state(cls, state, **kwargs):
        return hack_car_state(state, **kwargs)

    # signal should be 'left' or 'right' or None
    def set_turn_signal(self, signal):
        assert signal is not None
        self.turn_signal = signal

    def reset_turn_signal(self):
        self.turn_signal = None

    def get_intention(self):
        return self.intention

    def set_debug_intention(self, intention):
        self.intention = intention

    def set_intention_slow(self):
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

        set_seed(self.supervisor.game.map.seed, self.supervisor.game.time)
        choice = np.random.choice(np.where(scores == np.max(scores))[0])
        self.intention = all_ctrls[choice]
        self.spec_struct_trace = spec_struct_trace

    # only works when used with specification structure controller
    def set_intention(self):
        scores = []
        all_ctrls = self.get_all_ctrl()
        spec_struct_trace = od()

        best_ctrls = []
        best_score = 0
        for ctrl in all_ctrls:
            skipped = False # whether this ctrl action should be skipped
            max_possible_score = self.controller.specification_structure.max_score
            score = 0
            scores_sv = od()
            for oracle in self.controller.specification_structure.sorted_oracle_set:
                o_score = oracle.evaluate(ctrl, self, self.supervisor.game)
                o_tier = self.controller.specification_structure.tier[oracle]
                if oracle.name != 'backup_plan_safety':
                    score += int(o_score) * self.controller.specification_structure.tier_weights[o_tier]
                    max_possible_score -= (1-int(o_score))*self.controller.specification_structure.tier_weights[o_tier]
                if max_possible_score < best_score:
                    skipped = True
                    break

            if not skipped:
                if score == best_score:
                    scores.append(score)
                    best_ctrls.append(ctrl)
                elif score > best_score:
                    best_score = score
                    scores = [score]
                    best_ctrls = [ctrl]
                else:
                    print('error! this section should never be entered')

                scores_sv['total'] = score
                spec_struct_trace[ctrl_dict_to_tuple(ctrl)] = scores_sv

        set_seed(self.supervisor.game.map.seed, self.supervisor.game.time)
        choice = np.random.choice(best_ctrls)
        self.intention = choice
        self.spec_struct_trace = spec_struct_trace

    def set_token_count(self, cnt):
        self.token_count = cnt

    # reset conflict send and receive lists
    def reset_conflict_lists(self):
        self.agent_max_braking_not_enough = None
        self.send_conflict_requests_to = []
        self.received_conflict_requests_from = []
        self.conflict_winner = None
        self.is_winner = None

        self.received_sv = []
        self.sent_sv = []
        self.agents_checked_for_conflict_sv = []
        self.agent_max_braking_not_enough_sv = None
        self.conflict_winner_sv = None
        #self.clearance_straight_info = None
        #self.token_count_sv = 0

    # conflict resolution, returns True if winner and false if not winner
    def check_conflict_resolution_winner(self):
        # collect all agents in send and receive requests and find the winner
        conflict_cluster = self.send_conflict_requests_to + self.received_conflict_requests_from + [self]
        max_val = 0
        max_agent_list = []
        for agent in conflict_cluster:
            if agent.token_count > max_val:
                max_agent_list = [agent]
                # set max_val
                max_val = agent.token_count
            elif agent.token_count == max_val:
                max_agent_list.append(agent)
        # resolve ties with max values wifth agent ID comparison
        ind_max = np.argmax(np.array([agent.get_id() for agent in max_agent_list]))
        agent_winner = max_agent_list[ind_max]

        if len(conflict_cluster) > 1:
            return agent_winner.get_id() == self.get_id(), agent_winner
        # no other agents in conflict cluster
        else:
            return None, None

    def get_length_along_bundle(self):
        assert self.supervisor, 'Supervisory controller required!'
        x, y = self.state.x, self.state.y
        current_bundles = self.supervisor.game.map.tile_to_bundle_map[x,y]
        for bundle in current_bundles:
            if bundle.direction == self.state.heading:
                return bundle.tile_to_relative_length((x,y)), bundle

    # TODO: REDUNDANT use directed_tile_to_get_relative_width
    def get_width_along_bundle(self):
        x, y = self.state.x, self.state.y
        current_bundles = self.supervisor.game.map.tile_to_bundle_map[x,y]
        for bundle in current_bundles:
            if bundle.direction == self.state.heading:
                return bundle.tile_to_relative_width((x,y)), bundle

    def get_default_bubbles(self, front_back_only=False):
        # caching the agent bubbles
        def create_pickle_file(filename):
            vel = np.arange(self.v_min, self.v_max+1)
            bubble_dict = od()
            # for paper figure
            saved_bubble = od()
            for v in vel:
                bubble_dict[v], saved_bubble[v] = self.get_default_bubble(v, front_back_only=front_back_only)
            with open(filename, 'wb+') as pckl_file:
                pickle.dump(bubble_dict, pckl_file)
            with open(filename[:-2]+'_saved.p', 'wb+') as pckl_file:
                pickle.dump(saved_bubble, pckl_file)

        # string assumes v_min and a_min are negative
        car_param = 'v_' + 'n' + str(abs(self.v_min)) + '_' + str(self.v_max) + \
            '_a_n' + str(abs(self.a_min)) + '_' + str(self.a_max)+'.p'
        # check if file exists
        output_dir = os.getcwd()+'/saved_bubbles/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filedir = output_dir + car_param #v_n1_2_a_n2_2.p'
        if front_back_only:
            filedir = filedir[:-2]+'_filtered.p'
        if not os.path.exists(filedir):
            create_pickle_file(filedir)

        with open(filedir, 'rb') as pckl_file:
            data = pickle.load(pckl_file)
        return data

    # assuming backup plan is to brake maximally
    def get_maximum_braking_controls(self):
        def at_complete_stop(state):
            return state.v == 0
        plan = od()
        plan['controls'] = Car.get_ctrl_od(self.a_min, 'straight')
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
        return convert_car_orientation(inp)

    @classmethod
    def get_all_class_ctrl(cls, state, acc_vals, inverse=False, straight_only=False):
        all_ctrl = []
        for acc in acc_vals:
            if state.v + acc == 1 and not straight_only:
                steer_vals = ['straight', 'left-lane', 'right-lane', 'left-turn', 'right-turn']
            else:
                steer_vals  = ['straight']
            for steer in steer_vals:
                all_ctrl.append(Car.get_ctrl_od(acc, steer))
        return all_ctrl

    @classmethod
    def query_class_occupancy(cls, ctrl, state, v_min, v_max, inverse=False):
        return query_car_occupancy(ctrl, state, v_min, v_max, inverse=inverse)

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
                    #print("agent out of bounds, setting to have higher prec?")
                    #print(agent_tile)
                    agent_score = np.inf
                if agent_score > ego_score:
                    higher_pred.append(agent)
            return higher_pred
        agents_in_bubble = self.agents_in_bubble
        return get_agents_with_higher_precedence(agents_in_bubble)

    # check farthest straight agent can go forward (assuming agent in front already took its turn)
    def get_max_forward_ctrl(self):
        lead_agent = self.find_lead_agent(inside_bubble=True)
        if lead_agent is None:
            ctrl = Car.get_ctrl_od(self.a_max, 'straight')
            return ctrl
        x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
        # try all acceleration values
        ctrl_acc = np.arange(self.a_max, self.a_min-1, -1)
        for acc_val in ctrl_acc:
            ctrl = Car.get_ctrl_od(acc_val,'straight')
            occ = self.query_occupancy(ctrl)
            intersection = self.check_occupancy_intersection(occ, [lead_agent.state])
            safe_chk = self.check_safe_config(self, lead_agent, st_1=occ[-1], st_2=lead_agent.state)
            if safe_chk and not intersection: return ctrl
        return self.get_backup_plan_ctrl()

    # get the best straight action agent can do
    def get_best_straight_action(self):
        def ctrl_dict_to_tuple(ctrl):
            return (ctrl['steer'], ctrl['acceleration'])
        #print("finding best straight action!!!")
        # evaluate all the straight actions with the oracle
        straight_action_eval = od()
        scores = []
        all_straight_ctrl = self.get_all_ctrl(straight_only=True)
        for ctrl in all_straight_ctrl:
            score_save = od()
            score = 0
            for oracle in self.controller.specification_structure.oracle_set:
                o_score = oracle.evaluate(ctrl, self, self.supervisor.game)
                #if oracle.name == 'backup_plan_safety':
                #    print("backup plan safety")
                #    print(o_score)
                o_tier = self.controller.specification_structure.tier[oracle]
                try:
                    score += int(o_score) * self.controller.specification_structure.tier_weights[o_tier]
                except:
                    pass
                score_save[oracle.name] = o_score
            scores.append(score)
            score_save['total'] = score
            straight_action_eval[ctrl_dict_to_tuple(ctrl)] = score_save

        #seed = 1111
        set_seed(self.supervisor.game.map.seed, self.supervisor.game.time)
        choice = np.random.choice(np.where(scores == np.max(scores))[0])
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

        self.token_count_before = self.token_count

        # switch statements that define the outcome of the decision tree
        agent_type = get_agent_type()
        # True means no conflict with agents in bubble with higher precedence
        bubble_chk = not check_conflict_with_higher_precedence_agents()
        # if no conflict, then the agent is a winner
        if self.is_winner is None:
            cluster_chk = True
        else:
            cluster_chk = self.is_winner
        chk_receive_request_from_winner = chk_receive_request_from_winning_agent(self.conflict_winner)
        max_braking_enough = self.agent_max_braking_not_enough is None

        flag_dict = od()
        flag_dict['agent_type'] = agent_type
        flag_dict['higher_prec_chk'] = bubble_chk
        flag_dict['resolution_winner'] = cluster_chk
        flag_dict['max_braking_enough'] = max_braking_enough
        self.action_selection_flags = flag_dict

        safety_oracle = self.controller.specification_structure.oracle_set[6]
        assert safety_oracle.name == 'backup_plan_safety'

        # save info about sending and receiving requests
        #print("======================EVALUATING ACTION FOR AGENT AT STATE ======================")
        #print(self.state)
        #print(self.action_selection_flags)
        #__import__('ipdb').set_trace(context=21)

        if not max_braking_enough:
            if (agent_type == 'both' or agent_type == 'receiver') and not cluster_chk:
                #print(1)
                ctrl = check_min_dec_yield_req(self.conflict_winner)
                self.token_count = self.token_count+1
            else:
                #print(2)
                ctrl = self.get_best_straight_action()
                self.token_count = self.token_count+1
        else:
            # list of all possible scenarios and what action to take
            if agent_type == 'none' and bubble_chk and max_braking_enough:
                #print(3)
                valid_action = safety_oracle.evaluate(self.intention, self, self.supervisor.game)
                if valid_action:
                    ctrl = self.intention
                    self.token_count = 0
                else:
                    ctrl = self.get_best_straight_action()
                    self.token_count = self.token_count+1

            elif agent_type == 'none' and not bubble_chk:
                # TODO: take straight action, best safe one that aligns with intention
                #print(4)
                self.token_count = self.token_count+1
                ctrl = self.get_best_straight_action()
            elif agent_type == 'sender' and not cluster_chk:
                #print(5)
                # TODO: take straight action, best safe one that aligns with intention
                self.token_count = self.token_count+1
                ctrl = self.get_best_straight_action()
            elif agent_type == 'sender' and cluster_chk and not bubble_chk:
                #print(6)
                # TODO: take straight action, best safe one that aligns with intention
                self.token_count = self.token_count+1
                ctrl = self.get_best_straight_action()
            elif agent_type == 'sender' and cluster_chk and bubble_chk:
                #print(7)
                valid_action = safety_oracle.evaluate(self.intention, self, self.supervisor.game)
                if valid_action:
                    ctrl = self.intention
                    self.token_count = 0
                else:
                    ctrl = self.get_best_straight_action()
                    self.token_count = self.token_count+1
            elif (agent_type == 'receiver' or agent_type == 'both') and not cluster_chk:
                #print(8)
                # yield as much as needed for conflict winner to move
                # assumes winner has already taken its action!!!
                ctrl = check_min_dec_yield_req(self.conflict_winner)
                self.token_count = self.token_count+1
            elif (agent_type == 'receiver' or agent_type =='both') and bubble_chk and cluster_chk:
                #print(9)
                valid_action = safety_oracle.evaluate(self.intention, self, self.supervisor.game)
                if valid_action:
                    ctrl = self.intention
                    self.token_count = 0
                else:
                    ctrl = self.get_best_straight_action()
                    self.token_count = self.token_count+1
            elif (agent_type == 'receiver' or agent_type =='both') and not bubble_chk and cluster_chk:
                #print(10)
                # TODO: take straight action, best safe one that aligns with intention
                ctrl = self.get_best_straight_action()
                self.token_count = self.token_count+1
            else:
                print(agent_type)
                print(bubble_chk)
                print(cluster_chk)
                print("Error: invalid combination of inputs to action selection strategy!")
                ctrl = None
        self.token_count_sv = self.token_count
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
        #if bubble == None:
        bubble = self.get_bubble()
        occupancy_dict = self.supervisor.game.occupancy_dict
        agents_in_bubble = [occupancy_dict[grid_pt] for grid_pt in \
            bubble if grid_pt in occupancy_dict]
        return agents_in_bubble

    #=== find agents in this agents' conflict cluster ===================#
    def find_agents_to_send_conflict_request(self):
        #__import__('ipdb').set_trace(context=21)
        send_requests_list = []
        # collect all agents in bubble
        agents_in_bubble = self.agents_in_bubble

        # if agent intention is to go straight, it shouldn't send a request
        if self.intention['steer'] == 'straight' or self.intention['steer']=='right-turn' or self.intention['steer'] == 'left-turn':
            return send_requests_list
        # which agents checking to send
        agents_checked_for_conflict = []

        # check whether agent is in conflict with other agents in its bubble
        for agent in agents_in_bubble:
            if agent.get_id() != self.get_id():
                # debugging information!!
                try:
                    lon1 = self.get_length_along_bundle()[0]
                except:
                    lon1 = None
                try:
                    lon2 = agent.get_length_along_bundle()[0]
                except:
                    lon2 = None
                agents_checked_for_conflict.append((agent, lon1, lon2, self.state.heading==agent.state.heading))
                # first check if agent is longitudinally equal or ahead of other agent
                try:
                    chk = (self.get_length_along_bundle()[0]-agent.get_length_along_bundle()[0])
                    chk_lon = chk>=0
                    #if chk == 0:
                except:
                    break
                if chk_lon and self.state.heading == agent.state.heading:
                    # one last check to see whether agents are in the same lane and the agent behind doesn't also want to switch lanes
                    same_lane_chk = self.check_same_lane(self.state, agent.state)
                    if (not same_lane_chk) or (same_lane_chk and agent.intention['steer'] == 'left-lane' or agent.intention['steer'] == 'right-lane'):
                        chk_to_send_request = self.check_to_send_conflict_request(agent)
                        # add in check to see if agents are currently in the same lane
                        if chk_to_send_request:
                            send_requests_list.append(agent)
                        chk_max_braking_not_enough, flag_a, flag_b = self.intention_bp_conflict(agent)
                        #print(chk_max_braking_not_enough, flag_a, flag_b)
                        if chk_max_braking_not_enough:
                            self.agent_max_braking_not_enough = agent
                            # save this info
                            self.agent_max_braking_not_enough_sv = (agent.state.__tuple__(), flag_a, flag_b)
                            return []
        if self.supervisor.game.save_debug_info: 
            self.agents_checked_for_conflict_sv = [(ag[0].state.__tuple__(), ag[0].intention, ag[0].token_count_before, ag[0].get_id(), ag[1], ag[2], ag[3]) for ag in agents_checked_for_conflict]
        return send_requests_list

    #======add in code for checking whether agent can go for right turn=========#
    def check_right_turn_is_clear(self, right_turn_ctrl):
        # get the bundle that agent is trying to turn into
        #right_turn_ctrl = {'acceleration': 1-self.state.v, 'right-turn'}
        next_st = self.query_occupancy(right_turn_ctrl)[-1]
        bundle = self.supervisor.game.map.get_bundle_from_directed_tile((next_st.x, next_st.y), next_st.heading)
#        if bundle is None:
#            TODO: figure why this is necessary...
        # collect all agents in agent bundle AND in agent bubble
        for agent in self.agents_in_bubble:
            # get the bundle the agent in the bubble is in
            agent_bundle = agent.supervisor.game.map.get_bundle_from_directed_tile((agent.state.x, agent.state.y), agent.state.heading)
            if agent_bundle is not None:
                # if agent passes these checks, then see whether the agents are in conflict
                if bundle.get_id() == agent_bundle.get_id() and agent.get_id() != self.get_id():
                    # check whether an agent takes max acc and self takes right turn are valid actions
                    max_acc_ctrl = Car.get_ctrl_od(agent.a_max, 'straight')
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
        agents_in_bubble = self.agents_in_bubble
        all_agent_gridpts = [(agent.state.x, agent.state.y) for agent in agents_in_bubble if agent.get_id()!=self.get_id()]
        # don't check the first gridpoint in occupancy if list is greater than one
        occ = self.query_occupancy(ctrl)
        if len(occ) > 1:
            occ = occ[1:]
        action_gridpts = [(state.x, state.y) for state in occ]
        gridpts_intersect = list(set(all_agent_gridpts) & set(action_gridpts))
        return len(gridpts_intersect) > 0

    #============verifying agent back-up plan invariance===================#
    # check for collision with occupancy dict
    def check_collision(self, ctrl):
        # collect all grid points from occupancy dict except for own agent
        all_agent_gridpts = [gridpt for gridpt, agent in
                self.supervisor.game.occupancy_dict.items() if
                agent.get_id()!=self.get_id()]# and agent in
                #self.supervisor.game.simulated_agents]
        occ = self.query_occupancy(ctrl)
        if occ is None:
            return True
        else:
            # remove first gridpoint if more than one state in occ grid
            occ = self.query_occupancy(ctrl)
            if len(occ) > 1:
                occ = occ[1:]
            action_gridpts = [(state.x, state.y) for state in occ]
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
        if occupancy_dict is None:
            occupancy_dict = self.supervisor.game.occupancy_dict
        is_safe = True
        for gridpt, agent in occupancy_dict.items():
            x, y, v = agent.state.x, agent.state.y, agent.state.v
            lead_agent = agent.find_lead_agent(inside_bubble=True)
            # record all agents without valid back-up plan
            agents_no_bp = []
            if lead_agent:
                x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
                gap_curr = ((x_a-x)**2 + (y_a-y)**2)**0.5
                # not safe if gap is not large enough for any one of the agents
                if (compute_gap_req_fast(lead_agent.a_min, v_a, self.a_min, v) > gap_curr):
                    is_safe = False
                    agents_no_bp.append(agent.state.__tuple__())

        if return_list:
            return agents_no_bp
        else:
            return is_safe

    def find_lead_agent(self, state=None, inside_bubble=False, must_not_be_in_intersection=False, same_heading_required=True):
        if state is None:
            state = self.state

        # if only looking for lead agent inside of bubble
        if inside_bubble:
            # keep track of the closest lead agent
            closest_agent = None
            min_dist = np.inf
            for agent in self.agents_in_bubble:
                # check if agent is in front
                chk_ahead = check_agent_is_ahead(state.x, state.y, state.heading, agent.state.x, agent.state.y)
                if chk_ahead:
                    dx = agent.state.x-state.x
                    dy = agent.state.y-state.y
                    norm = math.sqrt(dx**2 + dy**2)
                    if (agent.get_id() != self.get_id()) and (not same_heading_required or (agent.state.heading == state.heading)):
                        if not must_not_be_in_intersection:
                            if norm < min_dist:
                                closest_agent = agent
                                min_dist = norm
                        else:
                            if not self.supervisor.game.map.tile_is_in_intersection((agent.state.x, agent.state.y)):
                                if norm < min_dist:
                                    closest_agent = agent
                                    min_dist = norm
            return closest_agent
        else:
            try:
                arc_l, bundle = self.get_length_along_bundle()
            except:
                return None
            d_vec = DIRECTION_TO_VECTOR[state.heading]
            # get tiles in front
#            tiles_x = np.arange(0,bundle.length-arc_l)*d_vec[0]+state.x
#            tiles_y = np.arange(0,bundle.length-arc_l)*d_vec[1]+state.y

            tiles_x = np.arange(0,min(FIND_LEAD_AGENT_THRESHOLD, bundle.length-arc_l))*d_vec[0]+state.x
            tiles_y = np.arange(0,min(FIND_LEAD_AGENT_THRESHOLD, bundle.length-arc_l))*d_vec[1]+state.y

            # check agents in these tiles
            for i in range(0, len(tiles_x)):
                # as soon as agent found in nearest tile, return lead vehicle
                if (tiles_x[i], tiles_y[i]) in self.supervisor.game.occupancy_dict:
                    agent = self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
                    # only find agents that are facing in the same direction
                    if (agent.get_id() != self.get_id()) and (not same_heading_required or (agent.state.heading == state.heading)):
                        if not must_not_be_in_intersection:
                            return self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
                        else:
                            if not self.supervisor.game.map.tile_is_in_intersection((agent.state.x, agent.state.y)):
                                return self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
            return None

    def compute_gap_req(self, lead_max_dec, lead_vel, follow_max_dec, follow_vel):
        #__import__('ipdb').set_trace(context=21)
        dx_lead = compute_dx_fast(lead_max_dec, lead_vel)
        dx_behind = compute_dx_fast(follow_max_dec, follow_vel)
        gap = max(dx_behind-dx_lead+1, 1)
        return gap

    # check if a set of actions is valid for a pair of agents
    def check_valid_actions(self, ag_1, ctrl_1, ag_2, ctrl_2, debug=False):
        #print("checking valid actions")
        #print(ag_1.state, ctrl_1, ag_2.state, ctrl_2)
        # get occupancy for both actions
        occ_1 = ag_1.query_occupancy(ctrl_1)
        occ_2 = ag_2.query_occupancy(ctrl_2)
        # if invalid actions, print an error
        if occ_1 is None or occ_2 is None:
            if debug:
                return False, 0, 0
            else:
                return False
        # check occupancy intersection
        chk_occupancy_intersection = self.check_occupancy_intersection(occ_1, occ_2)
        #print("occupancy check in check valid actions complete")
        chk_safe_end_config = self.check_safe_config(ag_1, ag_2, occ_1[-1], occ_2[-1])
        # return if occupancies don't intersect and safe end config
        if not debug:
            return (not chk_occupancy_intersection) and chk_safe_end_config
        else:
            return ((not chk_occupancy_intersection) and chk_safe_end_config), not chk_occupancy_intersection, chk_safe_end_config

    def check_same_lane(self, st_1, st_2):
        try:
            width_1, bundle_1 = self.supervisor.game.map.directed_tile_to_relative_width(((st_1.x, st_1.y), st_1.heading))
        except:
            return False
        try:
            width_2, bundle_2 = self.supervisor.game.map.directed_tile_to_relative_width(((st_2.x, st_2.y), st_2.heading))
        except:
            return False
            #print(st_1)
            ##print(st_2)
            #print(width_1, width_2)
        return bundle_1.get_id() == bundle_2.get_id() and width_1 == width_2


    # check if the final configuration of the agents is valid
    def check_safe_config(self, ag_1, ag_2, st_1=None, st_2=None):

        # returns agent_lead, agent_behind in that order
        def sort_agents(st_1, st_2):
            try:
                l_1, bundle_1 = self.supervisor.game.map.directed_tile_to_relative_length(((st_1.x, st_1.y), st_1.heading))
            except:
                return None, None, None, None
            try:
                l_2, bundle_2 = self.supervisor.game.map.directed_tile_to_relative_length(((st_2.x, st_2.y), st_2.heading))
            except:
                return None, None, None, None
            if l_1 > l_2:
                return ag_1, ag_2, st_1, st_2
            elif l_2 > l_1:
                return ag_2, ag_1, st_2, st_1
            else:
                return None, None, None, None

        if st_1 is None:
            st_1 = ag_1.state
        if st_2 is None:
            st_2 = ag_2.state

        #print("safe config check")
        #print(ag_1.state, ag_2.state, st_1, st_2)
        # first check same lane
        same_lane_chk = self.check_same_lane(st_1, st_2)
        #print('same lane check')
        #print(same_lane_chk)
        # TODO: if not in same lane, then agents are in safe config relative to each other?
        if not same_lane_chk:
            #print("not same lane check")
            return True
        # then check which agent is lead and which one is behind
        ag_lead, ag_behind, st_lead, st_behind = sort_agents(st_1, st_2)
        # if None, agents are on top of each other
        if ag_lead is None:
            #print("agents on top of each other ")
            #print()
            return False

        gap_req = compute_gap_req_fast(ag_lead.a_min, st_lead.v, ag_behind.a_min, st_behind.v)
        gap_curr = math.sqrt((st_lead.x-st_behind.x)**2+(st_lead.y-st_behind.y)**2)

        return gap_curr >= gap_req

    def check_occupancy_intersection(self, occ_a, occ_b):
        # convert list of agent states to grid points if not already list of tuples
        #print("checking occupancy intersection")
        if len(occ_a)>1:
            occ_a = occ_a[1:]
        if len(occ_b)>1:
            occ_b = occ_b[1:]
        if not isinstance(occ_a[0], tuple):
            occ_a = [(state.x, state.y) for state in occ_a]
        if not isinstance(occ_b[0], tuple):
            occ_b = [(state.x, state.y) for state in occ_b]
        occ_all = occ_a + occ_b
        if len(occ_all) != len(set(occ_all)):
            return True
        return False

    def get_backup_plan_ctrl(self, state=None):
        if state is None: state = self.state
        acc = self.a_min if state.v+self.a_min > 0 else -state.v
        return Car.get_ctrl_od(acc,'straight')

    # checks if maximal yield action by receiver is enough...
    def intention_bp_conflict(self, agent):
        if agent.state.heading == self.state.heading:
            chk_valid_actions, flag_a, flag_b = self.check_valid_actions(self, self.intention, agent, agent.get_backup_plan_ctrl(), debug=True)
            if not chk_valid_actions:
                print("max yield flag is set")
            return not chk_valid_actions, flag_a, flag_b
        else:
            return False, flag_a, flag_b

    #=== helper methods for computing whether to send conflict request to another agent =====#
    def check_to_send_conflict_request(self, agent):
        # check if gap between two agents is large enough for stopping!
        # TODO: see whether end state after these actions still have a back-up plan
        def intentions_conflict(agent):
            if agent.state.heading == self.state.heading:
                chk_valid_actions = self.check_valid_actions(self, self.intention, agent, agent.intention)
                if not chk_valid_actions:
                    if agent.state.heading == 'east':
                        #print(self.state)
                        print("sending conflict request")
                return not chk_valid_actions
            else:
                return False

        # should only send conflict requests to agents facing the same direction
        return intentions_conflict(agent) #and self.state.heading == agent.state.heading
        #or intention_forward_action_conflict(agent)

    #=== helper methods for computing the agent bubble ===================#
    def get_default_bubble(self, vel, front_back_only=False):
        def get_states_where_backup_plan_reaches_resources(resources):
            # brute force search for whether state reaches forward reachable gripts
            x_min = -self.v_max*5
            x_max = self.v_max*5
            x_grid = np.arange(x_min, x_max+1, 1)
            y_grid = np.arange(x_min, x_max+1, 1)
            v_grid = np.arange(self.v_min, self.v_max+1, 1)
            all_dir = ['north', 'south', 'east', 'west']
            bp_plan_states = []
            for x in x_grid:
                for y in y_grid:
                    for v in v_grid:
                        for heading in all_dir:
                            # check whether agent at this state will reach the forward reachable gridpoints if it executes the back-up plan
                            st = self.hack_state(self.state, x=x, y=y, heading=heading, v=v)
                            occ = self.query_occupancy(self.get_backup_plan_ctrl(), state=st)
                            if len(occ)>1:
                                occ = occ[1:]
                            gridpt_occ = [(state.x, state.y) for state in occ]
                            # check whether intersection between backup plan and resources
                            gridpts_intersect = list(set(resources) & set(gridpt_occ))
                            check_overlap = len(gridpts_intersect) > 1
                            # if overlap, append to list of states
                            if check_overlap:
                                bp_plan_states.append(st)
            return bp_plan_states


        # get all possible future reources car can occupy
        def get_resources():
            # set car state to have velocity v
            st = self.hack_state(self.state, x=self.default_state.x, y=self.default_state.y, heading=self.default_state.heading, v=vel)
            resources = []
            bubb_0 = []
            bubb_1 = []
            # make fake state here
            for ctrl in self.get_all_ctrl(state=st):
                #__import__('ipdb').set_trace(context=21)
                #print(ctrl)
                occ = self.query_occupancy(ctrl, state=st, inverse=False)
                #[print(state.x, state.y) for state in occ]
                if occ is not None:
                    resources_to_add = [(state.x, state.y) for state in occ]
                    resources.extend(resources_to_add)
                    bubb_0.extend(resources_to_add)
                    #print(resources)
                    # add in elements where car might need to execute emergency braking from the final state
                    final_st = occ[-1]
                    state_to_chk = self.hack_state(self.state, x=final_st.x, \
                        y=final_st.y, heading=final_st.heading, v=final_st.v)
                    safety_plan_resources = self.get_tiles_for_safety_plan(state=state_to_chk)
                    resources.extend(safety_plan_resources)
                    bubb_1.extend(safety_plan_resources)

            # remove any elements already inside
            resources_unique = []
            [resources_unique.append(x) for x in resources if x not in resources_unique]
            return resources_unique, list(set(bubb_0)), list(set(bubb_1))

        # gridpoints
        bubble = []
        resources, bubb_0, bubb_1 = get_resources()
        bubb_2 = []
        # get all possible reachable states to resources
        for xy in resources:
            # get all possible states to that resource
            # TODO: Right now, assuming all other agents have the same dynamics as itself!!
            states = self.get_backwards_reachable_states_from_gridpoint(xy, front_back_only=front_back_only)
            gridpts = [(state.x, state.y) for state in states]
            bubble.extend(gridpts)
            bubb_2.extend(gridpts)

        # get all states where backup plan intersects with resources
        backup_plan_states = get_states_where_backup_plan_reaches_resources(resources)
        # save the backup-plan states
        bubb_3 = []
        for state in backup_plan_states:
            bubb_3.append((state.x, state.y))

        # get all gridpoints from which the backup plan states are backwards reachable
        bubb_4 = []
        for state_to in backup_plan_states:
            states_from, actions_to = self.get_backwards_reachable_states(state_to, front_back_only=front_back_only)
            # transform all states from into gridpoints
            more_gridpts = [(state.x, state.y) for state in states_from]
            bubble.extend(more_gridpts)
            bubb_4.extend(more_gridpts)

        # remove repeat elements
        bubble_unique = []
        [bubble_unique.append(x) for x in bubble if x not in bubble_unique]
        bubb_2 = list(set(bubb_2))
        bubb_3 = list(set(bubb_3))
        bubb_4 = list(set(bubb_4))

        bubbles_sv = [bubb_0, bubb_1, bubb_2, bubb_4]
        # plot the bubble
        '''fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        for grid in bubble:
            rect = patches.Rectangle((grid[1],grid[0]),1,1,linewidth=0.5,facecolor='grey', alpha=0.3)
            ax.add_patch(rect)
        plt.show()'''

        return bubble_unique, bubbles_sv

    # compute number of tiles when applying brakes maximally
    def compute_dx(self, a_min, vel):
        dt = math.ceil(-vel/a_min)-1
        dx = int(np.sum([vel+(k+1)*a_min for k in range(dt)]))
        return dx

    # returns the tiles used in executing safety plan
    # note need agent to get max dec
    def get_tiles_for_safety_plan(self, state):
        dx = compute_dx_fast(self.a_min, state.v)
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
    def get_backwards_reachable_states(self, state_to, front_back_only=False):
        # just need a car object to call get_all_ctrl function
        filter_dir = [self.default_state.heading]
        all_inv_ctrl = self.get_all_ctrl(state=state_to, inverse=True)
        states = [None]*len(all_inv_ctrl) # will be too large but saves time pre-allocating
        actions = [None]*len(all_inv_ctrl)
        cnt = 0
        for inv_ctrl in all_inv_ctrl:
            occupancy = self.compute_state_from(state_to, inv_ctrl)
            if occupancy is not None:
                # remove all states without east west heading
                if front_back_only:
                    if occupancy.heading in filter_dir:
                        states[cnt] = occupancy
                        actions[cnt] = inv_ctrl
                        cnt = cnt+1
                else:
                    states[cnt] = occupancy
                    actions[cnt] = inv_ctrl
                    cnt = cnt+1
        # remove all the none
        states = states[0:cnt]
        actions = actions[0:cnt]
        return states, actions

    # get all backwards reachable states from a gridpoint
    def get_backwards_reachable_states_from_gridpoint(self, xy, front_back_only=False):
        all_directions = ['north', 'south', 'east', 'west']
        filter_dir = [self.default_state.heading]
        vel_arr = np.arange(self.v_min, self.v_max+1)
        backwards_reachable_states = []
        for dir in all_directions:
            for vel in vel_arr:
                # create a pseudo state
                state_to = self.hack_state(self.state, x=xy[0], y=xy[1], heading=dir, v=vel)
                states, _ = self.get_backwards_reachable_states(state_to)
                # filter for cars coming from certain direction
                if front_back_only:
                    states = [state for state in states if state.heading in filter_dir]
                backwards_reachable_states.extend(states)
        return backwards_reachable_states

    # transform the state with rotation and translation, do not assign heading
    # can be for tuple coordinate (x, y) or tuple state (x,y, heading, v)
    def transform_state(self, state, dx, dy, dtheta, assign_heading=False):
        # rotate and translate the vector
        arr = rotate_vector(tuple([state[0],state[1]]), dtheta*np.pi/180)
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
        if DEBUGGING:
            valid_joint_state = self.new_agent.check_joint_state_safety(occupancy_dict=occupancy_dict_chk)
        else:
            valid_joint_state = True
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

class Simulation:
    def __init__(self, the_map, agent_set=[]):
        self.map = the_map
        self.time = 0
        self.agent_set = agent_set
        self.draw_sets = [self.map.drivable_tiles, self.agent_set] # list ordering determines draw ordering
        self.occupancy_dict = od()
        self.update_occupancy_dict()

    def update_occupancy_dict(self):
        occupancy_dict = od()
        for agent in self.agent_set:
            x, y = agent.state.x, agent.state.y
            occupancy_dict[x,y] = agent
        self.occupancy_dict = occupancy_dict

    def update_occupancy_dict_for_one_agent(self, agent, prior_state, delete=False):
        if not delete:
            if prior_state is not None:
                self.occupancy_dict.pop((prior_state.x, prior_state.y), None)
            self.occupancy_dict[agent.state.x, agent.state.y] = agent
        else:
            self.occupancy_dict.pop((agent.state.x, agent.state.y), None)


    def play(self, t_end=np.inf, outfile=None):
        write_bool = outfile is not None and t_end is not np.inf
        while self.time < t_end:
            print("TIME: " + str(self.time))
            self.play_step()

    def time_forward(self):
        self.time += 1

    def sys_step(self):
        run(self.agent_set)

    def env_step(self):
        pass

    def play_step(self):
        self.sys_step()
        self.env_step()
        self.time_forward()

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
                # draw world and agents
                for draw_set in self.draw_sets:
                    artist.draw_set(draw_set)
                # update states
                self.play_step()
                # draw objects
                stdscr.refresh()
                time.sleep(frequency)
        finally:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()

class ContractGame(Simulation):
    def __init__(self, the_map, agent_set=[]):
        super(ContractGame, self).__init__(the_map, agent_set)

    # requires contract supervisor
    def learn(self):
        agent_to_legal_action_map = od()
        agent_set = list(self.agent_set)
        agent_to_signature_map = dict()
        for agent in agent_set:
            signature, mask = agent.supervisor.contract.frame.get_signature(game=agent.supervisor.game,
                                                                      agent=agent)
            xy_to_feature_map = dict(zip(mask, signature))
            agent_to_signature_map[agent] = signature, xy_to_feature_map
            if signature in agent.supervisor.contract.contract_draft:
                # if signature already has been encountered
                forbidden_actions = agent.supervisor.contract[signature]
            else:
                forbidden_actions = []
                for act in agent.get_all_ctrl():
                    next_state = agent.query_next_state(act)
                    next_tile = next_state.x, next_state.y
                    assert next_tile in xy_to_feature_map # the frame must satisfy this assertion
                    if xy_to_feature_map[next_tile] == 'out':
                        forbidden_actions.append(act)
                agent.supervisor.contract[signature] = forbidden_actions
            agent_to_legal_action_map[agent] = tuple(set(agent.get_all_ctrl()) - set(forbidden_actions))

        action_set = [agent_to_legal_action_map[agent] for agent in agent_set]
        action_product_set = itertools.product(*action_set) # note ordering is not preserved here

        existential_dictionary = dict()
        bad_joint_actions = []
        for joint_action in action_product_set:
            next_tiles = []
            is_bad_joint_action = False
            for agent_idx, agent_action in enumerate(joint_action):
                agent_next_state = agent_set[agent_idx].query_next_state(agent_action)
                agent_next_xy = agent_next_state.x, agent_next_state.y
                if agent_next_xy not in next_tiles:
                    next_tiles.append(agent_next_xy)
                else: # collision will happen
                    bad_joint_actions.append(joint_action)
                    is_bad_joint_action = True
                    break
            # collect (existential) proofs of good actions
            if not is_bad_joint_action:
                for agent_idx, agent_action in enumerate(joint_action):
                    agent = agent_set[agent_idx]
                    # get the signature to compute the quotient
                    agent_signature, _ = agent_to_signature_map[agent]
                    if agent_signature in existential_dictionary:
                        existential_dictionary[agent_signature].append(agent_action)
                    else:
                        existential_dictionary[agent_signature] = [agent_action]
            # forbid any action for which we have no proof of it being good
            for bad_joint_action in enumerate(bad_joint_actions):
                for agent_idx, agent_action in bad_joint_action:
                    agent = agent_set[agent_idx]
                    # get the signature to compute the quotient
                    agent_signature, _ = agent_to_signature_map[agent]
                    assert agent_signature in existential_dictionary # if this check fails, it's gameover
                    if agent_action in existential_dictionary[agent_signature]:
                        pass # this is a good action keep it
                    else:
                        agent.supervisor.contract[agent_signature] = [agent_action]
                        #TODO: this line must be executed, so if there are
                        #multiple proofs, it's ok to prune some to
                        #prevent this from not being entered

class TrafficGame(Simulation):
    """
    Traffic game class
    """
    # combines scenario + agents for game
    def __init__(self, game_map, save_debug_info=True, agent_set=[]):
        super(TrafficGame, self).__init__(the_map=game_map, agent_set=agent_set)
        self.draw_sets = [self.map.drivable_tiles, self.map.traffic_lights, self.agent_set] # list ordering determines draw ordering
        self.traces = od()
        self.traces['agent_ids'] = []
        self.traces_debug = od()
        self.occupancy_dict = od()
        self.update_occupancy_dict()
        self.car_count = 0
        self.collision_dict = od()
        self.out_of_bounds_dict = od()
        self.unsafe_joint_state_dict = od()
        self.save_debug_info = save_debug_info
        # save interesting numbers
        self.car_count = 0
        self.agents_reached_goal_count = 0 

    def spawn_agents(self):
        def valid_source_sink(source, sink):
            return not (source.node[0] == sink.node[0] or source.node[1] == sink.node[1])

        # generate random array
        set_seed(self.map.seed, self.time)
        # create two random arrays from this seed
        rand_arr1 = np.random.random(len(self.map.IO_map.sources))
        source = self.map.IO_map.sources[0]
        rand_arr2 = np.random.randint(len(self.map.IO_map.map[source]), size=len(rand_arr1))
        #print(rand_arr1)
        #print(rand_arr2)

        for i, source in enumerate(self.map.IO_map.sources):
            #print('source to check')
            #print(source.node)
            if rand_arr1[i] <= source.p:
                #print(len(self.map.IO_map.map[source]))
                sink = self.map.IO_map.map[source][rand_arr2[i]]
                # check if new car satisfies spawning safety contract
                #print('sources and sinks')
                #print(source.node, sink.node)
                new_car = create_default_car(source, sink, self, self.car_count)
                spawning_contract = SpawningContract(self, new_car)
                #print('state')
                #print(new_car.state)
                #print(spawning_contract.okay_to_spawn_flag)

                if spawning_contract.okay_to_spawn_flag:
                    #print(new_car.supervisor.goals)
                    self.agent_set.append(new_car)
                    self.update_occupancy_dict_for_one_agent(new_car,prior_state=None)
                    self.car_count = self.car_count+1

    def add_agent(self, agent):
        self.agent_set.append(agent)
    def save_plotting_info(self):
        lights = []
        agents = []
        # save all the agent states
        for agent in self.agent_set:
            spec_struct_trace = od()
            if self.time > 0:
                spec_struct_trace = agent.spec_struct_trace
            agents.append((agent.state.x, agent.state.y, \
                agent.state.heading, agent.state.v, agent.agent_color, agent.bubble, agent.get_id()))
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
        all_agent_info_at_time_t_dict = od()
        for agent in self.agent_set:
            # if prior state is not none
            prior_state = agent.prior_state
            if agent.prior_state is not None:
                prior_state = (agent.prior_state.x, agent.prior_state.y, agent.prior_state.heading, agent.prior_state.v)
                #print("final state of agent")
                #print(prior_state)
            # save all data in trace
            agent_trace_dict = {'subgoals': agent.supervisor.subgoals,
                                'state':prior_state,
                                'action': agent.ctrl_chosen,
                                'color':agent.agent_color,
                                'bubble':agent.bubble,
                                'goals': agent.supervisor.goals,
                                'spec_struct_info':agent.spec_struct_trace,
                                'agents_in_bubble': agent.agents_in_bubble_sv,
                                'sent': agent.sent_sv,
                                'received': agent.received_sv,
                                'token_count':agent.token_count_sv,
                                'max_braking_not_enough': agent.agent_max_braking_not_enough_sv,
                                'straight_action_eval': agent.straight_action_eval,
                                'action_selection_flags': agent.action_selection_flags,
                                'intention': agent.intention,
                                'conflict_winner': agent.conflict_winner_sv,
                                'token_count_before': agent.token_count_before,
                                'agent_id':agent.get_id(),
                                'left_turn_gap_arr':agent.left_turn_gap_arr,
                                'lead_agent':agent.lead_agent,
                                'checked_for_conflict':agent.agents_checked_for_conflict_sv,
                                'agents_in_bubble_before': agent.agents_in_bubble_before_sv,
                                'clearance_straight_info': agent.clearance_straight_info}
            all_agent_info_at_time_t_dict[agent.get_id()] = agent_trace_dict

        self.traces_debug[self.time] = all_agent_info_at_time_t_dict


    # write the game information to traces
    def write_game_info_to_traces(self, t_end):
        self.traces["map_name"] = self.map.map_name
        self.traces["spawn_probability"] = self.map.default_spawn_probability
        self.traces["collision_dict"] = self.collision_dict
        self.traces["out_of_bounds_dict"] = self.out_of_bounds_dict
        self.traces["unsafe_joint_state_dict"] = self.unsafe_joint_state_dict
        self.traces["t_end"] = t_end
        self.traces["special_heading_tiles"] = self.map.special_goal_tiles
        self.traces['seed'] = self.map.seed
        self.traces['total_agent_count'] = self.car_count
        self.traces['agents_reached_goal_count'] = self.agents_reached_goal_count
        self.traces['agents_in_map'] = len(self.agent_set)

    def write_data_to_pckl(self, filename, traces, new_entry=None):
        filename = filename + '.p'
        if new_entry is not None:
            traces.update(new_entry)
        with open(filename, 'wb+') as pckl_file:
            pickle.dump(traces, pckl_file)


    def sys_step(self):
        run(self.agent_set)

    def env_step(self):
        run(self.map.traffic_lights)
        self.spawn_agents()

    # check that all agents in the current config have a backup plan
    def check_config_safety(self):
        for agent in agent_set:
            x, y, v = agent.state.x, agent.state.y, agent.state.v
            lead_agent = plant.find_lead_agent(inside_bubble=True)
            if lead_agent:
                x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
                gap_curr = ((x_a-x)**2 + (y_a-y)**2)**0.5
                # not safe if gap is not large enough for any one of the agents
                if (compute_gap_req_fast(lead_agent.a_min, v_a, plant.a_min, v) >= gap_curr):
                    return False
        # all agents have backup plan
        return True

    # loop through all agents and send and receive conflict requests
    def send_and_receive_conflict_requests(self):
        # reset all requests
        for agent in self.agent_set:
            agent.reset_conflict_lists()

        for agent in self.agent_set:
            #print("========SENDING AND RECEIVING CONFLICT REQUESTS FOR: ======")
            #print(agent.state)
            # set list to send requests to
            agent.send_conflict_requests_to = agent.find_agents_to_send_conflict_request()
            # for each agent receiving request, update their receive list
            for agent_rec in agent.send_conflict_requests_to:
                agent_rec.received_conflict_requests_from.append(agent)

    def play(self, t_end=np.inf, outfile=None):
        write_bool = outfile is not None and t_end is not np.inf
        while self.time < t_end:
            print("TIME: " + str(self.time))
            # if save data to animate
            if write_bool:
                self.save_plotting_info()
            self.play_step()
            # need to save data after time step has occured
            if write_bool and self.time >= 0:
                if self.save_debug_info:
                    self.write_agents_to_traces()


        if write_bool:
            output_dir = os.getcwd()+'/saved_traces/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.write_game_info_to_traces(t_end)
            self.write_data_to_pckl(output_dir + outfile, self.traces)
            self.write_data_to_pckl(output_dir + outfile+'_debug', self.traces_debug)
        self.map.dump_cache()

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

class Field:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.grid = self.get_grid(self.csv_filename)
        self.nodes = list(self.grid.keys())
        self.drivable_tiles, self.drivable_nodes, self.non_drivable_nodes = self.get_drivable_tiles()

    def get_grid(self, csv_filename):
        grid = od()
        obstacles = od()
        with open(csv_filename + '.csv', 'rt') as f:
            graph = csv.reader(f)
            for i, row in enumerate(graph):
                for j, item in enumerate(row):
                    if item != 'x':
                        grid[i,j] = item
        return grid

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

class Map(Field):
    def __init__(self, csv_filename, default_spawn_probability=0.9, seed=None, random_traffic_lights_init=True):
        super(Map, self).__init__(csv_filename=csv_filename)
        self.map_name = csv_filename
        self.seed = seed
        self.default_spawn_probability = default_spawn_probability
        self.legal_orientations = self.get_legal_orientations()
        self.road_map = self.get_road_map()
        self.intersections = self.get_intersections()
        self.traffic_lights = self.get_traffic_lights(random_traffic_lights_init, self.seed)
        self.intersection_to_traffic_light_map = self.get_intersection_to_traffic_light_map()
        self.tile_to_intersection_map = self.get_tile_to_intersection_map()
        self.bundles = self.get_bundles()
        self.tile_to_bundle_map = self.get_tile_to_bundle_map()
        self.IO_map = self.get_IO_map()
        self.traffic_light_tile_to_bundle_map = self.get_traffic_light_tile_to_bundle_map()
        self.tile_to_traffic_light_map = self.get_tile_to_traffic_light_map()
        self.special_goal_tiles = []
        self.special_heading_tiles = []
        self.right_turn_tiles = self.find_right_turn_tiles()
        self.left_turn_tiles = self.find_left_turn_tiles()
        self.all_left_turns = self.get_all_left_turns()
        self.bundle_graph = self.get_bundle_graph()
        self.left_turn_to_opposing_traffic_bundles = self.get_left_turn_to_opposing_traffic_map()
        self.bundle_plan_cache = self.get_bundle_plan_cache()


    def tile_is_in_intersection(self, xy):
        """
        check if tile is in an intersection
        """
        try:
            num_orientations = len(self.legal_orientations[xy])
            return num_orientations > 1
        except:
            return False

    def get_bundle_plan_cache(self):
        path = os.getcwd() + '/saved_bundle_plans/' + self.map_name[7:]  + '.p'
        if os.path.exists(path):
            print('loading bundle plan cache from ' + path)
            with open(path, 'rb') as cache:
                cache = pickle.load(cache)
            bundle_plan_cache = cache
        else:
            bundle_plan_cache = od()
        return bundle_plan_cache

    def dump_cache(self):
        path = os.getcwd() + '/saved_bundle_plans/' + self.map_name[7:]  + '.p'
        print('dumping bundle plan cache to ' + path)
        with open(path, 'wb') as f:
            pickle.dump(self.bundle_plan_cache, f)
        print('done!')

    def directed_tile_to_relative_width(self, directed_tile):
        #__import__('ipdb').set_trace(context=21)
        x, y = directed_tile[0]
        heading = directed_tile[1]

        current_bundles = self.tile_to_bundle_map[x,y]
        for bundle in current_bundles:
            if bundle.direction == heading:
                return bundle.tile_to_relative_width((x,y)), bundle

    def directed_tile_to_relative_length(self, directed_tile):
        x, y = directed_tile[0]
        heading = directed_tile[1]

        current_bundles = self.tile_to_bundle_map[x,y]
        for bundle in current_bundles:
            if bundle.direction == heading:
                return bundle.tile_to_relative_length((x,y)), bundle

    # for now, we are assuming default car dynamics; TODO: generalize this
    def get_left_turn_to_opposing_traffic_map(self):
        left_turn_to_opposing_traffic_map = od()
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
                # relative occupancy contains the relative positions of the (left-turn) jumps on the opposing lane
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

        # bundle plan calculation
        signature = str(source) + str(sink)
        if signature in self.bundle_plan_cache:
            plan = self.bundle_plan_cache[signature]
        else:
            plan = nx.astar_path(planning_graph, source, sink)
            self.bundle_plan_cache[signature] = plan
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
        right = rotate_vector(tuple(forward), -np.pi/2)
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
        left = rotate_vector(tuple(forward), np.pi/2)
        next_tile = tuple(np.array(tile) + np.array(forward) + np.array(left))
        first_stop = (next_tile, next_direction)

        next_next_tile = tuple(np.array(next_tile) + np.array(left) + np.array(forward))
        second_stop = (next_next_tile, next_direction)
        try:
            next_bundle = self.directed_tile_to_bundle(second_stop)
            if next_bundle.is_leftmost_lane(next_next_tile):
                # update legal orientation dictionary
                self.legal_orientations[next_tile].append(next_direction)
                self.special_heading_tiles.append(first_stop)
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
            left = rotate_vector(tuple(forward), np.pi/2)
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
        right_turn_tiles = od()
        for bundle in self.bundles:
            right_turn_tiles[bundle] = od()
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

        left_turn_tiles = od()
        # goes through each bundle
        for bundle in self.bundles:
            # create a dictionary entry for the bundle
            left_turn_tiles[bundle] = od()
            direction = bundle.direction
            for idx in range(bundle.length):
                # pick tile in left-most lane
                tile = bundle.relative_coordinates_to_tile((bundle.width-1, idx))
                directed_tile = (tile, direction)
                # performs the check
                check, nxt = self.check_if_left_turn_tile(directed_tile)
                if check:
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
        tile_to_traffic_light_map = od()
        for traffic_light in self.traffic_lights:
            for tile in traffic_light.vtiles:
                tile_to_traffic_light_map[tile.xy] = traffic_light
            for tile in traffic_light.htiles:
                tile_to_traffic_light_map[tile.xy] = traffic_light
        return tile_to_traffic_light_map

    def get_traffic_light_tile_to_bundle_map(self):
        tile_to_bundle_map = od()
        for traffic_light in self.traffic_lights:
            for tile in traffic_light.vtiles:
                bundle = self.tile_to_bundle_map[tile.xy]
                append_or_create_new_list(tile_to_bundle_map, tile.xy, bundle)
            for tile in traffic_light.htiles:
                bundle = self.tile_to_bundle_map[tile.xy]
                append_or_create_new_list(tile_to_bundle_map, tile.xy, bundle)
        return tile_to_bundle_map

    def get_tile_to_bundle_map(self):
        tile_to_bundle_map = od()
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
        bins = od()
        all_bundles = []
        # initalizes bins to be empty lists
        for direction in DIRECTION_TO_VECTOR:
            bins[direction] = od()
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
                    cluster_projections = od()
                    for line in line_cluster:
                        for point in bins[direction][line]:
                            projection = point[1-direction_idx]
                            if projection not in cluster_projections:
                                cluster_projections[projection] = [line]
                            else:
                                cluster_projections[projection].append(line)
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

    def get_IO_map(self):
        sources, sinks = self.get_sources_sinks()
        IO_map = IOMap(sources=sources,sinks=sinks,map=od())
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

    def get_traffic_lights(self, random_traffic_lights_init, seed):
        def search_along(start, direction, search_length, green_symbol, yellow_symbol, red_symbol):
            nodes = []
            for k in range(search_length):
                node = tuple(np.array(start) + k*np.array(direction))
                if node in self.grid and self.grid[node] == green_symbol:
                    tl_node = TLNode(node, green_symbol, yellow_symbol, red_symbol)
                    nodes.append(tl_node)
            return nodes

        traffic_light_cnt = 0
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
                traffic_light = TrafficLight(count=traffic_light_cnt, light_id=light_id,htiles=htiles,vtiles=vtiles, random_init=random_traffic_lights_init, seed=seed)
                traffic_lights[traffic_light] = intersection
                traffic_light_cnt += 1

        return traffic_lights

    def get_intersection_to_traffic_light_map(self):
        intersection_to_traffic_light_map = od()
        for traffic_light in self.traffic_lights:
            intersection = self.traffic_lights[traffic_light]
            intersection_to_traffic_light_map[intersection] = traffic_light
        return intersection_to_traffic_light_map

    def get_tile_to_intersection_map(self):
        tile_to_intersection_map = od()
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
                dx, dy = rotate_vector(tuple(np.array([nex, ney])), rangle)
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

        legal_orientations = od()
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
    def __init__(self):
        pass
    def run_on(self, plant):
        raise NotImplementedError

class CompassController(Controller):
    def __init__(self):
        super(CompassController, self).__init__()
    def run_on(self, plant):
        plant.apply(random.choice([action for action in
            plant.get_all_ctrl() if plant]))

class RandomController(Controller):
    def __init__(self):
        super(RandomController, self).__init__()
    def run_on(self, plant):
        plant.apply(random.choice([action for action in
            plant.get_all_ctrl() if plant]))

class ObstacleAvoidingRandomController(Controller):
    def __init__(self):
        super(ObstacleAvoidingRandomController, self).__init__()
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

class IntersectionClearanceOracle(Oracle):
    def __init__(self):
        super(IntersectionClearanceOracle, self).__init__(name='intersection_clearance')
    def evaluate(self, ctrl_action, plant, game):
        # return count of number of agents ahead of it in interesection
        def count_agents_in_intersection_ahead(intersection_gap):
            cnt = 0
            # check for number of agents ahead in intersection
            forward = DIRECTION_TO_VECTOR[plant.state.heading]
            curr_st = np.array([plant.state.x, plant.state.y])
            for i in range(1, intersection_gap):
                next_tile_tuple = tuple(curr_st + i*np.array(forward))
                # if there is an agent there, then count it
                if next_tile_tuple in game.occupancy_dict:
                    cnt = cnt + 1
            return cnt

        self.clearance_straight_info = None
        current_state = plant.state.x, plant.state.y
        x_curr, y_curr = current_state
        next_state = plant.query_next_state(ctrl_action)
        x_next, y_next = next_state.x, next_state.y
        # need to check whether agent backup plan at next state will be in intersection
        bp_state = plant.query_occupancy(plant.get_backup_plan_ctrl(), state=next_state)[-1]
        x_next_bp, y_next_bp = bp_state.x, bp_state.y

        # check if not crossing into an intersection
        try:
            currently_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_curr,y_curr))
            will_be_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_next,y_next))
            bp_will_be_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_next_bp,y_next_bp))
        except:
            return True
        if currently_in_intersection or not will_be_in_intersection or not bp_will_be_in_intersection:
            return True
        else:
            #print("action crossing into intersection")
            # if indeed crossing into an intersection
            # check if attempting to perform a left turn
            current_subgoal = plant.supervisor.subgoals[0]
            # figure out what intersection is being entered
            next_intersection = game.map.tile_to_intersection_map[(x_next, y_next)]
            current_heading = plant.state.heading
            # confirm intention to perform left turn; TODO: generalize this check
            if plant.supervisor.game.map.tile_is_in_intersection((current_subgoal[0][0], current_subgoal[0][1])):
                # left turn is confirmed
                heading_degrees = Car.convert_orientation(current_heading)
                left_heading_degrees  = (heading_degrees + 90) % 360
                left_heading = Car.convert_orientation(left_heading_degrees)
                forward = DIRECTION_TO_VECTOR[current_heading]
                next_tile = tuple(np.array(forward) + np.array([x_curr, y_curr]))
                while left_heading not in plant.supervisor.game.map.legal_orientations[next_tile]:
                    next_tile = tuple(np.array(forward) + np.array([next_tile[0], next_tile[1]]))
                reference_state = plant.hack_state(plant.state, x=next_tile[0], y=next_tile[1], heading=left_heading)

                if current_heading in ['east', 'west']:
                    intersection_gap = next_intersection.height
                else:
                    intersection_gap = next_intersection.width
                lead_agent = plant.find_lead_agent(reference_state, must_not_be_in_intersection=True, same_heading_required=False)
                if lead_agent:
                    width, _= plant.get_width_along_bundle()
                    reference_state_bundle_width = width + 1
                    num_residual_tiles = intersection_gap - reference_state_bundle_width
                    clearance = max(abs(lead_agent.state.x-reference_state.x), abs(lead_agent.state.y-reference_state.y))
                else:
                    clearance = np.inf
                return clearance > intersection_gap #TODO: find a better bound

            else: # going straight
                #print("entering intersection straight with intention to move straight")
                # save intersection gap,
                if current_heading in ['east', 'west']:
                    intersection_gap = next_intersection.width
                else:
                    intersection_gap = next_intersection.height
                # count the number of agents in intersection ahead of it
                agent_cnt_in_intersection = count_agents_in_intersection_ahead(intersection_gap)
                lead_agent = plant.find_lead_agent(plant.state, must_not_be_in_intersection=True, same_heading_required=False)
                if lead_agent:
                    clearance = max(abs(lead_agent.state.x-x_curr), abs(lead_agent.state.y-y_curr)) - intersection_gap - agent_cnt_in_intersection
                else:
                    clearance = np.inf

                if lead_agent is None:
                    x_sv = None
                    y_sv = None
                else:
                    x_sv = lead_agent.state.x
                    y_sv = lead_agent.state.y

                self.clearance_straight_info = (x_curr, y_curr, x_sv, y_sv, intersection_gap, agent_cnt_in_intersection, clearance)
                return clearance > intersection_gap #TODO: find a better bound

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

#TODO: improve some calc here...

class OracleController(Controller):
    def __init__(self, oracle_set):
        super(OracleController, self).__init__()
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
    def __init__(self, specification_structure):
        super(SpecificationStructureController, self).__init__()
        self.specification_structure = specification_structure

    def manage_turn_signals(self, plant, ctrl):
        next_st = plant.query_occupancy(ctrl)[-1]
        x_nxt, y_nxt, heading_nxt = next_st.x, next_st.y, next_st.heading
        next_directed_tile = ((x_nxt,y_nxt), heading_nxt)
        # turn left signal on if ctrl ends up in first left turn tile and
        # agent subgoal is in left turn tiles
        if plant.supervisor.subgoals[0] in plant.supervisor.game.map.all_left_turns and \
            plant.supervisor.subgoals[0] == next_directed_tile:
            plant.set_turn_signal('left')
        # turn off signal when ctrl makes agent leave the intersection
        x, y = plant.state.x, plant.state.y
        if len(plant.supervisor.game.map.legal_orientations[x,y]) > 1 and len(plant.supervisor.game.map.legal_orientations[x_nxt,y_nxt]) == 1:
            plant.reset_turn_signal()

    def run_on(self, plant):
        # choose action according to action selection strategy
        ctrl = plant.action_selection_strategy()
        self.manage_turn_signals(plant, ctrl)
        #print("applying control")
        plant.apply(ctrl)
        # heading check to see if agent reached goal
        if (plant.state.x, plant.state.y, plant.state.heading) == plant.supervisor.goals: 
            plant.supervisor.game.agents_reached_goal_count += 1

class Supervisor():
    def _init__(self):
        self.plant = None
        self.game = None

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

class LocalContractSupervisor(Supervisor):
    def __init__(self, game, goal, contract):
        self.game = game
        self.goal = goal
        self.contract = contract

    def get_next_goal_and_plan(self):
        return self.goal, None

    def check_goals(self):
        pass

class GoalExit(Supervisor):
    def __init__(self, game, goals=None):
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

class GoalCycler(Supervisor):
    def __init__(self, game, goals=None):
        self.game = game
        self.goals = cycle(self.add_headings_to_goals(goals))

    def add_headings_to_goals(self, goals):
        new_goals = []
        for goal in goals:
            if len(goal) < 3:
                heading = self.game.map.legal_orientations[goal][0] # take first one if multiple
                new_goals.append((goal[0], goal[1], heading))
            else:
                new_goals.append(goal)
        return new_goals

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

class BundleGoalExit(Supervisor):
    def __init__(self, game, goals=None):
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
                self.game.update_occupancy_dict_for_one_agent(self.plant,prior_state=None,delete=True)

class SpecificationStructure():
    def __init__(self, oracle_list, oracle_tier):
        assert len(oracle_list) == len(oracle_tier), 'list of oracles must have the same length as corresponding list of tiers'
        self.oracle_set = oracle_list
        self.tier = self.set_tier(oracle_tier)
        self.tier_weights = self.set_tier_weights(oracle_tier)
        self.max_score = self.get_max_score()
        self.sorted_oracle_set = self.get_sorted_oracle_set()

    def get_sorted_oracle_set(self):
        sorted_oracle_set  = sorted(self.oracle_set, key=lambda oracle: self.tier[oracle])
        return sorted_oracle_set

    def get_max_score(self):
        score = 0
        for oracle in self.oracle_set:
            o_score = True
            o_tier = self.tier[oracle]
            if oracle.name != 'backup_plan_safety':
                score += int(o_score) * self.tier_weights[o_tier]
        return score

    def set_tier(self, oracle_tier):
        tier = od()
        for idx, oracle in enumerate(self.oracle_set):
            tier[oracle] = oracle_tier[idx]
        return tier

    def set_tier_weights(self, oracle_tier):
        def num(tier):
            return np.sum(np.array(oracle_tier) == tier)
        all_tiers = np.sort(list(set(oracle_tier)))[::-1]
        tier_weights = od()
        tier_weights[all_tiers[0]] = 1
        for idx in range(1, len(all_tiers)):
            tier = all_tiers[idx]
            next_lower_tier = all_tiers[idx-1]
            tier_weights[tier] = (num(next_lower_tier)+1)*tier_weights[next_lower_tier]
        return tier_weights

class TrafficLight:
    def __init__(self, light_id, htiles, vtiles, seed, count, t_green=20,t_yellow=3,t_buffer=10, random_init=False):
        self.id = light_id
        self.count = count
        self.durations = od()
        self.durations['green'] = t_green
        self.durations['yellow'] = t_yellow
        self.t_buffer = t_buffer
        self.durations['red'] = self.durations['green'] + self.durations['yellow'] + self.t_buffer * 2

        if random_init:
            #if seed is not None:
                #print(self.count)
                #print (seed + self.count)
            set_seed(seed,self.count*3)
            self.hstate = np.random.choice([color for color in self.durations])
            self.htimer = np.random.choice(self.durations[self.hstate])
        else:
            self.htimer = 0
            self.hstate = self.get_next_state(self.hstate)
        self.htiles = htiles
        self.vtiles = vtiles

    def get_next_state(self, state):
        if state == 'green':
            return 'yellow'
        elif state == 'yellow':
            return 'red'
        else:
            return 'green'

    def check_directed_light_in_N_turns(self, direction, N):
        future_lights = self.check_light_N_turns_from_now(N)
        if direction in ['east', 'west']:
            color, _ = future_lights['vertical']
        elif direction in ['north', 'south']:
            color, _ = future_lights['horizontal']
        return color

    def get_id(self):
        return id(self)

    def set_traffic_lights(self, hstate, htimer):
        #traffic_light = self.tile_to_traffic_light_map[traffic_tile]
        assert hstate in ['red', 'yellow', 'green']
        assert htimer < self.durations[hstate]
        self.hstate = hstate
        self.htimer = htimer

    def check_light_N_turns_from_now(self, N):
        hstate, htimer = self.ghost_run_N_time_steps(N)

        state = od()
        state['horizontal'] = hstate, htimer
        state['vertical'] = self.get_vstate_given_hstate(hstate, htimer)
        return state

    def ghost_run_N_time_steps(self, N):
        hstate = self.hstate
        htimer = self.htimer
        for i in range(N):
            htimer += 1
            if htimer >= self.durations[hstate]:
                hstate = self.get_next_state(hstate)
                htimer = 0
        return hstate, htimer

    def get_vstate_given_hstate(self, hstate, htimer):
        if hstate == 'green' or hstate == 'yellow':
            color = 'red'
            if hstate == 'green':
                timer = self.t_buffer + htimer
            elif hstate == 'yellow':
                timer = self.t_buffer + self.durations['green'] + htimer
        else: # if red
            if htimer < self.t_buffer:
                color = 'red'
                timer = htimer + self.durations['green'] + self.durations['yellow'] + self.t_buffer
            elif htimer < self.t_buffer + self.durations['green']:
                color = 'green'
                timer = htimer - self.t_buffer
            elif htimer < self.t_buffer + self.durations['green'] + self.durations['yellow']:
                color = 'yellow'
                timer = htimer - self.t_buffer - self.durations['green']
            else:
                color = 'red'
                timer = htimer - self.t_buffer - self.durations['green'] - self.durations['yellow']
        return color, timer

    def run(self):
        self.htimer += 1
        if self.htimer >= self.durations[self.hstate]:
            self.hstate = self.get_next_state(self.hstate)
            self.htimer = 0

    def get_hstate(self):
        return self.hstate, self.htimer

    def get_vstate(self):
        return self.get_vstate_given_hstate(self.hstate, self.htimer)

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
    intersection_clearance_oracle = IntersectionClearanceOracle()
    oracle_set = [static_obstacle_oracle,
                  traffic_light_oracle,
                  legal_orientation_oracle,
                  backup_plan_progress_oracle,
                  maintenance_progress_oracle,
                  improvement_progress_oracle,
                  backup_plan_safety_oracle,
                  unprotected_left_turn_oracle,
                  traffic_intersection_oracle,
                  intersection_clearance_oracle] # type: List[Oracle]
    specification_structure = SpecificationStructure(oracle_set, [1, 2, 2, 3, 4, 4, 1, 1, 2, 2])
    return specification_structure

def create_default_car(source, sink, game, car_count):
    ss = get_default_car_ss()
    spec_struct_controller = SpecificationStructureController(specification_structure=ss)
    start = source.node
    end = sink.node
    car = Car(x=start[0],y=start[1],heading=start[2],v=0,v_min=0,v_max=3, a_min=-1,a_max=1, car_count=car_count, seed=game.map.seed)
    car.set_controller(spec_struct_controller)
    supervisor = BundleGoalExit(game=game, goals=[end])
    car.set_supervisor(supervisor)
    return car

def create_specified_car(attributes, game):
    def parse_goal_string_to_goal_tuple(string):
        string=string.replace(" ", "")
        string=string.replace("(", "")
        string=string.replace(")", "")
        x, y, direction = string.split(',')
        return (int(x), int(y), direction)

    if ('goal' not in attributes) or (attributes['goal'] == 'auto'):
        set_seed(game.map.seed)
        attributes['goal'] = np.random.choice(game.map.IO_map.sinks).node
    else:
        attributes['goal'] = parse_goal_string_to_goal_tuple(attributes['goal'])
    if 'controller' not in attributes:
        ss = get_default_car_ss()
        # default to SpecificationStructureController
        controller = SpecificationStructureController(specification_structure=ss)
        attributes['controller'] = controller
    if 'agent_color' not in attributes:
        # if not specified choose a random color
        agent_color = np.random.choice(CAR_COLORS)
        attributes['color'] = color
    if 'v_min' not in attributes:
        attributes['v_min'] = 0
    if 'v_max' not in attributes:
        attributes['v_max'] = 3
    if 'a_min' not in attributes:
        attributes['a_min'] = -1
    if 'a_max' not in attributes:
        attributes['a_max'] = 1
    if 'heading' not in attributes or attributes['heading'] == 'auto':
        attributes['heading'] = game.map.legal_orientations[(attributes['x'],
                                attributes['y'])][0]
    if 'v' not in attributes:
        attributes['v'] = 0
    car =  Car(x=attributes['x'], y=attributes['y'],
            heading=attributes['heading'], v=attributes['v'],
            v_min=attributes['v_min'], v_max=attributes['v_max'],
            a_min=attributes['a_min'],a_max=attributes['a_max'], car_count=0)
    # set car color
    car.agent_color = attributes['agent_color']
    car.set_controller(attributes['controller'])
    supervisor = BundleGoalExit(game=game, goals=[attributes['goal']])
    car.set_supervisor(supervisor)
    return car

class QuasiSimultaneousGame(TrafficGame):
    def __init__(self, game_map, save_debug_info=True):
        super(QuasiSimultaneousGame, self).__init__(game_map=game_map, save_debug_info=save_debug_info)
        self.bundle_to_agent_precedence = self.get_bundle_to_agent_precedence()
        self.bundle_to_agent_precedence = None
        self.simulated_agents = []

    def done_simulating_agent(self, agent):
        """
        add agent to list of simulated agent list
        """
        self.simulated_agents.append(agent)

    def done_simulating_everyone(self):
        self.simulated_agents = []

    def get_bundle_to_agent_precedence(self):
        bundle_to_agent_precedence = od()
        for bundle in self.map.bundles:
            bundle_to_agent_precedence[bundle] = od()
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

    # do a global collision check and add to traces the overlapping gridpoints at
    # the right time step
    def global_collision_check(self, all_occupancy_gridpts):
        # find out all elements that are duplicates and return list of duplicates
        dup = [pt for pt in all_occupancy_gridpts if all_occupancy_gridpts.count(pt) > 1]
        # saves traces
        try:
            self.traces["global_traces"][self.time] = dup
        except:
            self.traces["global_traces"] = od()
            self.traces["global_traces"][self.time] = dup

        # returns true if collision occurs
        if len(dup) > 0:
            "collision has occurred"
            print(self.time)
            print(set(dup))
        return len(dup) > 0

    def set_agent_bubbles(self):
        for agent in self.agent_set:
            agent.bubble = agent.get_bubble()
            agent.agents_in_bubble = agent.find_agents_in_bubble()
            # write to trace format
            if agent.supervisor.game.save_debug_info: 
                agent.agents_in_bubble_before_sv = [[agent.state.__tuple__(), agent.agent_color, agent.get_id()] for agent in agent.agents_in_bubble]
        pass

    def set_agent_intentions(self):
        for agent in self.agent_set:
            agent.set_intention()

    def resolve_precedence(self):
        self.bundle_to_agent_precedence = self.get_bundle_to_agent_precedence()

    def determine_conflict_cluster_resolutions(self):
        for agent in self.agent_set:
            if agent.supervisor.game.save_debug_info: 
                agent.sent_sv = [(ag.state.__tuple__(), ag.intention, ag.token_count, ag.get_id()) for ag in agent.send_conflict_requests_to]
                agent.received_sv = [(ag.state.__tuple__(), ag.intention, ag.token_count, ag.get_id()) for ag in agent.received_conflict_requests_from]
            agent.is_winner, agent.conflict_winner = agent.check_conflict_resolution_winner()
            if agent.supervisor.game.save_debug_info: 
                if agent.conflict_winner is not None:
                    agent.conflict_winner_sv = agent.conflict_winner.state.__tuple__()

    def sys_step(self):
        # set all agent intentions
        self.set_agent_intentions()
        self.set_agent_bubbles()
        # call resolve_conflicts
        self.send_and_receive_conflict_requests()
        self.determine_conflict_cluster_resolutions()
        # write send and receive conflict requests to file
        # resolve precedence
        self.resolve_precedence()
        all_occupancy_gridpts = []
        for bundle in self.map.bundles:
            precedence_list = list(self.bundle_to_agent_precedence[bundle].keys())
            precedence_list.sort(reverse=True)
            for precedence in precedence_list:
                for agent in self.bundle_to_agent_precedence[bundle][precedence]:
                    # reset agents in bubble list bc of quasi-game
                    agent.agents_in_bubble = agent.find_agents_in_bubble()
                    if agent.supervisor.game.save_debug_info: 
                        agent.agents_in_bubble_sv = [[agent.state.__tuple__(), agent.agent_color, agent.get_id()] for agent in agent.agents_in_bubble]
                    state = agent.state
                    agent.run()
                    # add in occupancy of agent when it took its action
                    #print(agent.ctrl_chosen)
                    occ = agent.query_occupancy(agent.ctrl_chosen, state=state)
                    if len(occ) > 1:
                        occ = occ[1:]
                    gridpts = [(state.x, state.y) for state in occ]
                    all_occupancy_gridpts.extend(gridpts)
                    self.done_simulating_agent(agent)
        self.done_simulating_everyone()

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
    
    print("total agents spawned")
    print(traces['total_agent_count'])

    print("total agents that reached goal")
    print(traces['agents_reached_goal_count'])
    
    print("agents currently in map at end of game")
    print(traces['agents_in_map'])


def parse_config(csv_file_path, config_file_path):
    """
    parse configuration file
    """
    # get agents from csv mask
    def get_agents():
        agents = od()
        with open(csv_file_path, 'rt') as f:
            graph = csv.reader(f)
            for i, row in enumerate(graph):
                for j, item in enumerate(row):
                    if item in AGENT_CHAR:
                        agents[item] = (i, j)
        return agents
    # get agents' attributes from config file
    def get_attributes():
        agents = od()
        data = import_json(config_file_path)
        return data
    xy_agents = get_agents()
    attrs_agents = get_attributes()
    data_all_agents = od()
    for agent in xy_agents:
        x, y = xy_agents[agent]
        data_all_agents[agent] = od()
        data_all_agents[agent]['x'] = x
        data_all_agents[agent]['y'] = y
        for attr in attrs_agents[agent]:
            data_all_agents[agent][attr] = attrs_agents[agent][attr]
    return data_all_agents

def import_json(infile):
    with open(infile) as f:
        data = json.load(f, object_pairs_hook=od)
    return data

def create_qs_game_from_config(game_map, config_path):
    # create game object
    game = QuasiSimultaneousGame(game_map=the_map)
    # create csv config file path
    csv_file_path = config_path + '.csv'
    # create json config file path
    config_file_path = config_path + '.json'
    configs = parse_config(csv_file_path, config_file_path)
    for agent in configs:
        new_car = create_specified_car(configs[agent], game)
        game.agent_set.append(new_car)
    game.update_occupancy_dict()
    return game

if __name__ == '__main__':
    seed = 1205
    map_name = 'straight_road'
    the_map = Map('./maps/'+map_name,default_spawn_probability=0.35, seed=seed)
    output_filename = 'game'

    # create a game from map/initial config files
    game = QuasiSimultaneousGame(game_map=the_map)
    #game = create_qs_game_from_config(game_map=the_map, config_path='./configs/'+map_name)

    # play or animate a normal game
    game.play(outfile=output_filename, t_end=2)
    #game.animate(frequency=0.01)

    # print debug info
    debug_filename = os.getcwd()+'/saved_traces/'+ output_filename + '.p'
    print_debug_info(debug_filename)

    # play debugged game

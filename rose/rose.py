'''

    File name: rose.py
    Authors: Tung Phan, Karena Cai
    Date created: 1/10/2020
'''
from typing import List, Any
from itertools import cycle
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
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.array([int(round(x)) for x in np.matmul(rot_mat, vec)])

class Drawable:
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
        self.state = self.query_next_state(ctrl)
        self.supervisor.game.update_occupancy_dict()

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
        # attributes relating to conflict stuff
        # TODO: change intention usingg intention generator
        actions = [{"acceleration": 0, "steer": 'right-lane'}, {"acceleration": 0, "steer":'straight'}, {'acceleration':0, 'steer':'left-lane'}]
        self.intention = random.choice(actions)
        self.send_conflict_requests_to = [] # list of agents
        self.received_conflict_requests_from = [] # list of agents
        self.token_count = 0

    def set_intention(self, action):
        self.intention = action
    
    def set_token_count(self, cnt):
        self.token_count = cnt

    # reset conflict send and receive lists 
    def reset_conflict_lists(self):
        self.send_conflict_reqeusts_to = []
        self.received_conflict_requests_from = []
    
    # conflict resolution, returns True if winner and false if not winner
    def check_conflict_resolution_winner(self):
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
        return agent_winner.get_id() == self.get_id()
    # update token count according to which action agent took and whether 
    # it aligns with agent intention
    def update_token_count():
        pass

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
            print(vel)
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
        step = 1
        tile_sequence_chain = []
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
    def get_all_class_ctrl(cls, state, acc_vals, inverse=False):
        all_ctrl = []
        def make_ctrl(acc, steer):
            return {'steer': steer, 'acceleration': acc}
        for acc in acc_vals:
            if state.v + acc == 1:
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

    def get_all_ctrl(self, state=None,inverse=False):
        if state is None: state = self.state
        return Car.get_all_class_ctrl(state, self.acc_vals, inverse=inverse)

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
        send_requests_list = []
        # collect all agents in bubble
        agents_in_bubble = self.find_agents_in_bubble()
        # if agent intention is to go straight, it shouldn't send a request
        if self.intention['steer'] == 'straight': return send_requests_list

        # check whether agent is in conflict with other agents in its bubble
        for agent in agents_in_bubble:
            if agent.get_id() != self.get_id():
                chk_to_send_request = self.check_to_send_conflict_request(agent)
                if chk_to_send_request: send_requests_list.append(agent)

        return send_requests_list
    
    #============verifying agent back-up plan invariance===================#
    def find_lead_agent(self, state=None):
        if state is None: state = self.state
        arc_l, bundle = self.get_length_along_bundle()
        d_vec = DIRECTION_TO_VECTOR[state.heading]
        # get tiles in front
        tiles_x = np.arange(0,bundle.length-arc_l)*d_vec[0]+state.x
        tiles_y = np.arange(0,bundle.length-arc_l)*d_vec[1]+state.y

        # check agents in these tiles
        for i in range(0, len(tiles_x)): 
            # as soon as agent found in nearest tile, return lead vehicle
            if (tiles_x[i], tiles_y[i]) in self.supervisor.game.occupancy_dict: 
                agent = self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
                if agent.state.heading == self.state.heading and agent.get_id() != self.get_id(): 
                    return self.supervisor.game.occupancy_dict[(tiles_x[i], tiles_y[i])]
        return None
    
    
    def compute_gap_req(self, lead_max_dec, lead_vel, follow_max_dec, follow_vel):
        #__import__('ipdb').set_trace(context=21)
        dx_lead = self.compute_dx(lead_max_dec, lead_vel)
        dx_behind = self.compute_dx(follow_max_dec, follow_vel)
        gap = max(dx_behind-dx_lead+1, 1)
        return gap

    #=== helper methods for computing whether to send conflict request to another agent =====#
    def check_to_send_conflict_request(self, agent):
        # check if gap between two agents is large enough for stopping!
        def check_safe_config(ag_1, ag_2, st_1=None, st_2=None): 
            if st_1 is None: st_1 = ag_1.state
            if st_2 is None: st_2 = ag_2.state

            # check agents are in the same lane
            def check_same_lane(ag_1, ag_2): 
                #__import__('ipdb').set_trace(context=21)
                width_1, bundle_1 = ag_1.get_width_along_bundle()
                width_2, bundle_2 = ag_2.get_width_along_bundle()
                return bundle_1.get_id() == bundle_2.get_id() and width_1 == width_2
            
            # returns agent_lead, agent_behind in that order
            def sort_agents(ag_1, ag_2):
                l_1 = ag_1.get_length_along_bundle()[0]
                l_2 = ag_2.get_length_along_bundle()[0]
                if l_1 > l_2:
                    return ag_1, ag_2, st_1, st_2
                elif l_2 > l_1:
                    return ag_2, ag_1, st_2, st_1
                else: 
                    return None, None
            
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

        def occupancy_intersection(occ_a, occ_b):
            # convert list of agent states to grid points if not already list of tuples
            if len(occ_a)>1: occ_a = occ_a[1:]
            if len(occ_b)>1: occ_b = occ_b[1:]
            if not isinstance(occ_a[0], tuple): occ_a = [(state.x, state.y) for state in occ_a]
            if not isinstance(occ_b[0], tuple): occ_b = [(state.x, state.y) for state in occ_b]
            occ_all = occ_a + occ_b
            if len(occ_all) != len(set(occ_all)):
                return True
            return False

        # TODO: see whether end state after these actions still have a back-up plan
        def intentions_conflict(agent):
            occ_a = self.query_occupancy(ctrl=self.intention)
            occ_b = agent.query_occupancy(ctrl=agent.intention)
            if occ_a is None or occ_b is None: 
                print("error: agent intention not allowed")
                return True
            chk_safe_end_state_config = check_safe_config(self, agent, st_1=occ_a[-1], st_2=occ_b[-1])
            return occupancy_intersection(occ_a, occ_b) or not chk_safe_end_state_config
    
        # see whether agent intention conflicts with another agent's back-up plan
        def intention_bp_conflict(agent):
            # get acceleration needed to come to a stop (if not enough, maximal)
            def get_backup_plan_ctrl(agent, state): 
                acc = agent.a_min if state.v+agent.a_min > 0 else -state.v 
                return {'acceleration':acc, 'steer':'straight'}

            occ_a = self.query_occupancy(ctrl=self.intention)
            occ_b = agent.query_occupancy(ctrl=get_backup_plan_ctrl(agent, agent.state))
            if occ_a is None or occ_b is None: 
                print("error: intention is not allowed")
                return True

            chk_safe_end_state_config = check_safe_config(self, agent, st_1=occ_a[-1], st_2=occ_b[-1])
            return occupancy_intersection(occ_a, occ_b) or not chk_safe_end_state_config

        # first check if agent is longitudinally equal or ahead of other agent
        chk_lon = (self.get_length_along_bundle()[0]-agent.get_length_along_bundle()[0])>=0
        if not chk_lon: return False

        return intentions_conflict(agent) or intention_bp_conflict(agent)


    #=== helper methods for computing the agent bubble ===================#
    def get_default_bubble(self, vel):
        # get all possible future reources car can occupy
        def get_resources():
            # set car state to have velocity v
            st = self.hack_state(self.state, x=self.default_state.x, y=self.default_state.y, heading=self.default_state.heading, v=vel)
            resources = []
            # make fake state here
            for ctrl in self.get_all_ctrl():
                #print(ctrl)
                occ = self.query_occupancy(ctrl, state=st, inverse=False)
                if occ is not None: 
                    resources_to_add = [(state.x, state.y) for state in occ]
                    resources.extend(resources_to_add)
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

class Game:
    # combines scenario + agents for game
    def __init__(self, game_map, agent_set=[]):
        self.time = 0
        self.map = game_map
        self.agent_set = agent_set
        self.draw_sets = [self.map.drivable_tiles, self.map.traffic_lights, self.agent_set] # list ordering determines draw ordering
        self.occupancy_dict = dict()
        self.update_occupancy_dict()

    def update_occupancy_dict(self):
        occupancy_dict = dict()
        for agent in self.agent_set:
            x, y = agent.state.x, agent.state.y
            occupancy_dict[x,y] = agent
        self.occupancy_dict = occupancy_dict

    def spawn_agents(self):
        for source in self.map.IO_map.sources:
            if np.random.uniform() <= source.p:
                sink = np.random.choice(self.map.IO_map.map[source])
                new_car = create_default_car(source, sink, self)
                self.agent_set.append(new_car)
    def add_agent(self, agent):
        self.agent_set.append(agent)
    def time_forward(self):
        self.time += 1
    def save_snapshot(self):
        lights = []
        agents = []
        # save all the agent states
        for agent in self.agent_set:
            agents.append((agent.state.x, agent.state.y, \
                agent.state.heading, agent.agent_color, agent.get_bubble()))
        # save all the traffic light states
        for traffic_light in self.map.traffic_lights:
            for tile in traffic_light.htiles:
                x, y = tile.xy
                lights.append((x, y, traffic_light.get_hstate()[0]))
            for tile in traffic_light.vtiles:
                x, y = tile.xy
                lights.append((x, y, traffic_light.get_vstate()[0]))
        return {"lights": lights, "agents": agents}
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
                for node in self.map.drivable_nodes:
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
                game.time_forward()
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

    # check that all agents in the current config have a backup plan
    def check_config_safety(self):
        for agent in agent_set():
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
        traces = dict()
        while self.time < t_end:
            print("TIME: " + str(self.time))
            # if save data to animate
            if write_bool:
                snapshot = self.save_snapshot()
                traces[self.time] = snapshot
            #self.check_conflict_requests_karena_debug()

            self.play_step()
            game.time_forward()

        if write_bool:
            output_dir = os.getcwd()+'/saved_traces/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.write_data_to_pckl(output_dir + outfile, traces)

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
    def __init__(self, csv_filename, default_spawn_probability=0, random_traffic_lights_init=False):
        self.grid = self.get_grid(csv_filename)
        self.default_spawn_probability = default_spawn_probability
        self.drivable_nodes = list(self.grid.keys())
        self.drivable_tiles, self.non_drivable_nodes = self.get_drivable_tiles()
        self.legal_orientations = self.get_legal_orientations()
        self.road_map = self.get_road_map()
        self.intersections = self.get_intersections()
        self.traffic_lights = self.get_traffic_lights(random_traffic_lights_init)
        self.bundles = self.get_bundles()
        self.tile_to_bundle_map = self.get_tile_to_bundle_map()
        self.IO_map = self.get_IO_map()
        self.traffic_light_tile_to_bundle_map = self.get_traffic_light_tile_to_bundle_map()
        self.tile_to_traffic_light_map = self.get_tile_to_traffic_light_map()
        self.right_turn_tiles = self.find_right_turn_tiles()
        self.left_turn_tiles = self.find_left_turn_tiles()
        self.bundle_graph = self.get_bundle_graph()
#        self.directed_tile_to_turns(((23,9), 'east'))
#        print(self.get_bundle_plan(((23,9), 'east'), ((49, 88), 'north')))

    def get_bundle_plan(self, source, sink):
        planning_graph = self.bundle_graph.copy()
        original_edges = list(planning_graph.edges)

        for edge in original_edges:
            end_node = edge[1]
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
        tile, direction = directed_tile
        bundle = self.directed_tile_to_bundle(tile, direction)
        rel_tile = bundle.tile_to_relative_position(tile)
        return rel_tile

    def check_directed_tile_reachability(self, dtile_start, dtile_final):
        if not isinstance(dtile_start[1], str):
            dtile_start = dtile_start[1]
        if not isinstance(dtile_final[1], str):
            dtile_final = dtile_final[0]

        bundle_start = self.directed_tile_to_bundle(dtile_start[0], dtile_start[1])
        bundle_final = self.directed_tile_to_bundle(dtile_final[0], dtile_final[1])
        if bundle_start != bundle_final:
            return False
        else:
            rel_tile_start = self.directed_tile_to_relative_bundle_tile(dtile_start)
            rel_tile_final = self.directed_tile_to_relative_bundle_tile(dtile_final)
            length_diff = rel_tile_final[1] - rel_tile_start[1]
            width_diff = abs(rel_tile_final[0]- rel_tile_start[0])
            return length_diff > width_diff

    def directed_tile_to_turns(self, directed_tile):
        turns = []
        bundle = self.directed_tile_to_bundle(directed_tile[0], directed_tile[1])

        for turn in self.right_turn_tiles[bundle]:
            if self.check_directed_tile_reachability(directed_tile, turn):
                turns.append(turn)

        for turn in self.left_turn_tiles[bundle]:
            precrossing_tile = self.get_precrossing_left_turn_tile(turn)
            if precrossing_tile is None:
                turn_node = turn
                precrossing_tile = turn_node
            else:
                turn_node = (precrossing_tile, turn)
            if self.check_directed_tile_reachability(directed_tile, precrossing_tile):
                turns.append(turn_node)
        return turns

    def get_bundle_graph(self):
        '''
        constructs bundle graph

        '''
        bundle_graph = nx.DiGraph()
        for bundle in self.right_turn_tiles:
            for from_tile in self.right_turn_tiles[bundle]:
                to_tile = self.right_turn_tiles[bundle][from_tile]
                bundle_graph.add_edge(from_tile, to_tile)
                turns = self.directed_tile_to_turns(to_tile)
                for turn in turns:
                    bundle_graph.add_edge(to_tile, turn)

        for bundle in self.left_turn_tiles:
            for from_tile in self.left_turn_tiles[bundle]:
                to_tile = self.left_turn_tiles[bundle][from_tile]

                precrossing_tile = self.get_precrossing_left_turn_tile(from_tile)
                if precrossing_tile:
                    bundle_graph.add_edge((precrossing_tile, from_tile), to_tile)
                else:
                    bundle_graph.add_edge(from_tile, to_tile)
                turns = self.directed_tile_to_turns(to_tile)
                for turn in turns:
                    bundle_graph.add_edge(to_tile, turn)

        return bundle_graph

    def get_precrossing_left_turn_tile(self, left_turn_tile):
        tile_xy, tile_direction = left_turn_tile
        backward = -np.array(DIRECTION_TO_VECTOR[tile_direction])
        new_tile = tuple(np.array(tile_xy)+backward)
        while True:
            if new_tile not in self.legal_orientations or self.legal_orientations[new_tile] is None:
                return None
            elif len(self.legal_orientations[new_tile]) > 1:
                new_tile = tuple(np.array(new_tile)+backward)
            else:
                return new_tile, tile_direction

    def check_if_right_turn_tile(self, tile, direction):
        assert direction in self.legal_orientations[tile]
        direction_degrees = Car.convert_orientation(direction)
        next_direction_degrees = (direction_degrees - 90)%360
        next_direction = Car.convert_orientation(next_direction_degrees)
        forward = DIRECTION_TO_VECTOR[direction]
        right = rotate_vector(forward, -np.pi/2)
        next_tile = tuple(np.array(tile) + np.array(forward) + np.array(right))
        try:
            next_bundle = self.directed_tile_to_bundle(next_tile, next_direction)
            return next_bundle.is_rightmost_lane(next_tile), (next_tile, next_direction)
        except:
            return False, None

    def check_if_left_turn_tile(self, tile, direction):
        assert direction in self.legal_orientations[tile]
        direction_degrees = Car.convert_orientation(direction)
        next_direction_degrees = (direction_degrees + 90)%360
        next_direction = Car.convert_orientation(next_direction_degrees)
        forward = DIRECTION_TO_VECTOR[direction]
        left = rotate_vector(forward, np.pi/2)
        next_tile = tuple(np.array(tile) + np.array(forward) + np.array(left))
        try:
            next_bundle = self.directed_tile_to_bundle(next_tile, next_direction)
            return next_bundle.is_leftmost_lane(next_tile), (next_tile, next_direction)
        except:
            return False, None

    # assuming agents can only legally make a right turn from the rightmost lane into rightmost lane
    def find_right_turn_tiles(self):
        right_turn_tiles = dict()
        for bundle in self.bundles:
            right_turn_tiles[bundle] = dict()
            direction = bundle.direction
            for idx in range(bundle.length):
                tile = bundle.relative_coordinates_to_tile((0, idx))
                check, nxt = self.check_if_right_turn_tile(tile, direction)
                if check:
                    right_turn_tiles[bundle][(tile, direction)] = nxt
        return right_turn_tiles

    # assuming agents can only legally make a left turn from the leftmost lane into leftmost lane
    def find_left_turn_tiles(self):
        left_turn_tiles = dict()
        for bundle in self.bundles:
            left_turn_tiles[bundle] = dict()
            direction = bundle.direction
            for idx in range(bundle.length):
                tile = bundle.relative_coordinates_to_tile((bundle.width-1, idx))
                check, nxt = self.check_if_left_turn_tile(tile, direction)
                if check:
                    left_turn_tiles[bundle][(tile, direction)] = nxt
        return left_turn_tiles

    def directed_tile_to_bundle(self, tile, heading=None):
        assert tile in self.tile_to_bundle_map, 'Tile does not belong to any bundle!'
        bundles = self.tile_to_bundle_map[tile]
        if heading is None:
            assert len(bundles) == 1
            bundle = bundles[0]
        else:
            bundle_idx = np.nonzero([b.direction == heading for b in bundles])[0][0]
            bundle = bundles[bundle_idx]
        return bundle

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
        for xy in self.drivable_nodes:
            if self.grid[xy] == '-':
                symbol = '.'
                non_drivable_nodes.append(xy)
            else:
                symbol = self.grid[xy]
            tile = DrivableTile(xy,symbol)
            drivable_tiles.append(tile)
        return drivable_tiles, non_drivable_nodes

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

        traffic_lights = []
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
                traffic_lights.append(traffic_light)

        return traffic_lights

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

        road_cells = [cell for cell in self.drivable_nodes if self.legal_orientations[cell]]

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
    def __init__(self):
        pass
    def evaluate(self, ctrl_action, plant, game):
        raise NotImplementedError

class ReplanProgressOracle(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(ReplanProgressOracle, self).__init__()
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
        super(PathProgressOracle, self).__init__()
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
        super(BundleProgressOracle, self).__init__()
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

class TrafficLightOracle(Oracle):
    def __init__(self):
        super(TrafficLightOracle, self).__init__()

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
        return action_not_running_a_red_light and backup_plant_will_still_be_ok

#TODO: improve some calc here...
class BackupPlanSafetyOracle(Oracle):
    def __init__(self):
        super(BackupPlanSafetyOracle, self).__init__()
    def evaluate(self, ctrl_action, plant, game):
        next_state = plant.query_occupancy(ctrl_action)[-1]
        x, y, heading, v = next_state.x, next_state.y, next_state.heading, next_state.v
        lead_agent = plant.find_lead_agent(state=next_state)

        if lead_agent:
            x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
            gap_curr = ((x_a-x)**2 + (y_a-y)**2)**0.5
            return plant.compute_gap_req(lead_agent.a_min, v_a, plant.a_min, v) <= gap_curr
        else:
            return True

class StaticObstacleOracle(Oracle):
    def __init__(self):
        super(StaticObstacleOracle, self).__init__()
    def evaluate(self, ctrl_action, plant, game):
        return all([(occ_state.x, occ_state.y) in game.map.drivable_nodes
            for occ_state in plant.query_occupancy(ctrl_action)])

class LegalOrientationOracle(Oracle):
    def __init__(self):
        super(LegalOrientationOracle, self).__init__()
    def evaluate(self, ctrl_action, plant, game):
        final_state = plant.query_occupancy(ctrl_action)[-1]
        final_node = final_state.x, final_state.y
        legal_headings = None
        if final_node in game.map.drivable_nodes:
            legal_headings = game.map.legal_orientations[final_node]
        return legal_headings and final_state.heading in game.map.legal_orientations[final_node]

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
        #print("choosing action for agent " + str(plant.state.__tuple__()))
        #print(plant.supervisor.game.occupancy_dict.keys())
        scores = []
        all_ctrls = plant.get_all_ctrl()
        for ctrl in all_ctrls:
            score = 0
            for oracle in self.specification_structure.oracle_set:
                o_score = oracle.evaluate(ctrl, plant, self.game)
                o_tier = self.specification_structure.tier[oracle]
                try:
                    score += int(o_score) * self.specification_structure.tier_weights[o_tier]
                except:
                    pass
            scores.append(score)

        choice = random.choice(np.where(scores == np.max(scores))[0])
        plant.apply(all_ctrls[choice])

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
        self.current_bundle_plan = self.get_current_bundle_plan()

    def get_current_bundle_plan(self):
        bundles = []
        for node in self.current_plan:
            x,y,heading = node
            bundle = self.game.map.tile_to_bundle_map[x,y]
            bundles.append(bundle)
        separated_bundles = [grouped[0] for grouped in separate_list(bundles, 'equal')]

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
        next_plan = nx.astar_path(game.map.road_map, source, target)
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
        next_plan = nx.astar_path(game.map.road_map, source, target)
        return next_goal, next_plan

    def check_goals(self):
        if self.plant:
            if np.sum(np.abs(np.array([self.plant.state.x, self.plant.state.y]) - np.array([self.current_goal[0], self.current_goal[1]]))) == 0: # if close enough
                self.current_goal, self.current_plan = self.get_next_goal_and_plan()

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
    def __init__(self, light_id, htiles, vtiles, t_green=20,t_yellow=3,t_buffer=4, random_init=True):
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
    progress_oracle = BundleProgressOracle()
    oracle_set = [static_obstacle_oracle, traffic_light_oracle, legal_orientation_oracle, progress_oracle, backup_plan_safety_oracle] # type: List[Oracle]
    specification_structure = SpecificationStructure(oracle_set, [1, 1, 2, 3, 1])
    return specification_structure

def create_default_car(source, sink, game):
    ss = get_default_car_ss()
    spec_struct_controller = SpecificationStructureController(game=game,specification_structure=ss)
    start = source.node
    end = sink.node
    car = Car(x=start[0],y=start[1],heading=start[2],v=0,v_min=0,v_max=2, a_min=-2,a_max=2)
    car.set_controller(spec_struct_controller)
    supervisor = GoalExit(game=game, goals=[end])
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
            bundles = self.map.tile_to_bundle_map[(x,y)]
            for bundle in bundles:
                if bundle.direction == heading:
                    longitudinal_precedence = bundle.tile_to_relative_length((x,y))
                    append_or_create_new_list(bundle_to_agent_precedence[bundle], \
                                              longitudinal_precedence, \
                                              agent)
        return bundle_to_agent_precedence

    def resolve_precedence(self):
        self.bundle_to_agent_precedence = self.get_bundle_to_agent_precedence()

    def sys_step(self):
        self.resolve_precedence()
        for bundle in self.map.bundles:
            precedence_list = list(self.bundle_to_agent_precedence[bundle].keys())
            precedence_list.sort(reverse=True)
            for precedence in precedence_list:
                for agent in self.bundle_to_agent_precedence[bundle][precedence]:
                    agent.run()

    def play_step(self):
        self.sys_step()
        self.env_step()

if __name__ == '__main__':
#    the_map = Map('./maps/straight_road', default_spawn_probability=0.001)
    the_map = Map('./maps/city_blocks', default_spawn_probability=0.01)
    output_filename = 'game.p'

    game = QuasiSimultaneousGame(game_map=the_map)
#    game.play(outfile=output_filename, t_end=100)
    game.animate(frequency=0.01)

    #game = Game(game_map=the_map)
    #num_agents = 5
    #play_fixed_agent_game_karena_debug(num_agents, game)

    #def world_changer(func):
    #    def world_changing_function(*args, **kwargs):
    #        func(*args, **kwargs)
    #        plant = args[1]
    #        plant.supervisor.game.update_occupancy_dict()
    #    return world_changing_function

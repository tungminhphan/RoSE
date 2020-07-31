from global_constants import *
import math
import copy as cp
import numpy as np

CAR_OCCUPANCY_DICT = {}

class Memoize:
    """
    caching class intended to be used as a decorator
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        # check if args is memoizable
        signature = str(args)
        # check cache
        if signature not in self.memo:
            self.memo[signature] = self.fn(*args)
        return self.memo[signature]

@Memoize
def compute_gap_req_fast(lead_max_dec, lead_vel, follow_max_dec, follow_vel):
    dx_lead = compute_dx_fast(lead_max_dec, lead_vel)
    dx_behind = compute_dx_fast(follow_max_dec, follow_vel)
    gap = max(dx_behind-dx_lead+1, 1)
    return gap

@Memoize
def compute_dx_fast(a_min, vel):
    """
    memoized version of compute_dx
    """
    dt = math.ceil(-vel/a_min)-1
    dx = int(sum([vel+(k+1)*a_min for k in range(dt)]))
    return dx

@Memoize
def rotate_vector(vec, theta):
    """
    take in a 2D vector and an angle in radians and outputs the same vector
    rotated by the amount specified by the input angle in a CCW fashion.
    """
    # create rotation matrix
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.array([int(round(x)) for x in np.matmul(rot_mat, vec)])

def rotate_matrix(mat, theta):
    """
    take in a 2D matrix and an angle in radians and outputs the same vector
    rotated by the amount specified by the input angle in a CCW fashion.
    """
    # create rotation matrix
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(rot_mat, mat)

def ctrl_dict_to_tuple(ctrl):
    return (ctrl['steer'], ctrl['acceleration'])

def hack_car_state(state, **kwargs):
    new_state = cp.copy(state)
    if 'x' in kwargs:
        new_state.x = kwargs.get('x')
    if 'y' in kwargs:
        new_state.y = kwargs.get('y')
    if 'heading' in kwargs:
        new_state.heading = kwargs.get('heading')
    if 'v' in kwargs:
        new_state.v = kwargs.get('v')
    return new_state

def query_car_occupancy(ctrl, state, v_min, v_max, inverse):
    signature = str(ctrl) + str(state) + str(v_min) + \
                str(v_max) + str(inverse)
    if signature in CAR_OCCUPANCY_DICT:
        return CAR_OCCUPANCY_DICT[signature]
    else:
        acc = ctrl['acceleration']
        steer = ctrl['steer']

        def get_new_heading(act):
            heading = convert_car_orientation(state.heading)
            if act == 'left-turn':
                heading += 90
            elif act == 'right-turn':
                heading -= 90
            return convert_car_orientation(heading)

        def relativize_absolute_displacement(abs_dis, inverse=False):
            rel_dis = []
            offset = np.array([state.x, state.y])
            for displacement in abs_dis:
                heading = convert_car_orientation(state.heading) * np.pi/180
                rel_dis.append(offset + rotate_vector(tuple(displacement), heading))
                offset_reverse = rotate_vector(tuple(displacement), heading)
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
            abs_displacements = [[0, 0], [0, 1], [-1, 1]]
        elif steer == 'left-turn' and final_v == 1:
            abs_displacements = [[0, 0], [0, 1], [-1, 1]]
            final_heading = get_new_heading('left-turn')
        elif steer == 'right-lane' and final_v == 1:
            abs_displacements = [[0, 0], [0, 1], [1, 1]]
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
            occupancy_list[i] = hack_car_state(state, x=xy[0], y=xy[1], heading=heading, v=vel)

        CAR_OCCUPANCY_DICT[signature] = occupancy_list
        return occupancy_list

def check_agent_is_ahead(agent_x, agent_y, agent_heading, other_agent_x, other_agent_y):
    dx = other_agent_x-agent_x
    dy = other_agent_y-agent_y

    if agent_heading in ['north', 'south']:
        if dy != 0:
            is_ahead = False
        elif agent_heading == 'north':
            is_ahead = dx < 0
        else: # agent facing south
            is_ahead = dx > 0
    else: # facing east/west
        if dx != 0:
            is_ahead = False
        elif agent_heading == 'east':
            is_ahead = dy > 0
        else: # agent facing west
            is_ahead = dy < 0
    return is_ahead

def convert_car_orientation(inp):
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

class Oracle():
    def __init__(self, name):
        self.name = name
        pass
    def evaluate(self, ctrl_action, plant, game):
        raise NotImplementedError

class StaticObstacleOracle(Oracle):
    def __init__(self):
        super(StaticObstacleOracle, self).__init__(name='static_obstacle')
    def evaluate(self, ctrl_action, plant, game):
        # check if action is safe
        next_occupancy = plant.query_occupancy(ctrl_action)
        for occ_state in next_occupancy:
            if not ((occ_state.x, occ_state.y) in game.map.drivable_nodes):
                return False
        else:
        # check if backup plan would be safe
            next_state = next_occupancy[-1]
            tile_sequence_chain = plant.query_backup_plan(state=next_state)
            tiles = set([tile for tiles in [tile_turn[-1]
                for tile_turn in tile_sequence_chain] for tile in
                tiles])
            for tile in tiles:
                if not (tile in game.map.drivable_nodes):
                    return False
            return True

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

    def check_if_light_red_in_N_turns(self, traffic_light, direction, N):
        color = traffic_light.check_directed_light_in_N_turns(direction, N)
        return color == 'red'

    def action_not_running_a_red_light(self, ctrl_action, plant, game):
        occ_states = plant.query_occupancy(ctrl_action)
        occ_tiles = [(state.x, state.y) for state in occ_states]
        return self.tile_sequence_not_running_a_red_light_on_N_turn(occ_tiles, game, N=1)

    def check_if_crossing(self, light_tile, tiles, direction):
        if direction in ['west', 'east']:
            k = 1
        elif direction in ['north', 'south']:
            k = 0
        light_tile_projection = light_tile[k]
        tile_projections = [tile[k] for tile in tiles]
        if direction in ['east', 'south']:
            return min(tile_projections) <= light_tile_projection and max(tile_projections) > light_tile_projection
        elif direction in ['west', 'north']:
            return min(tile_projections) < light_tile_projection and max(tile_projections) >= light_tile_projection

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
            try:
                traffic_light = game.map.tile_to_traffic_light_map[(plant.state.x, plant.state.y)]
                light_is_red = self.check_if_light_red_in_N_turns(traffic_light, plant.state.heading, 0) # N=0
            except:
                light_is_red = True # if no traffic light, assume it's red
            if light_is_red:
                # check if right turn is valid
                return plant.check_right_turn_is_clear(ctrl_action)
            else:
                return action_not_running_a_red_light and backup_plant_will_still_be_ok
        else:
            return action_not_running_a_red_light and backup_plant_will_still_be_ok

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
            # if indeed crossing into an intersection
            # check if attempting to perform a left turn
            current_subgoal = plant.supervisor.subgoals[0]
            # figure out what intersection is being entered
            next_intersection = game.map.tile_to_intersection_map[(x_next, y_next)]
            current_heading = plant.state.heading
            # confirm intention to perform left turn; TODO: generalize this check
            if plant.supervisor.game.map.tile_is_in_intersection((current_subgoal[0][0], current_subgoal[0][1])):
                # left turn is confirmed
                heading_degrees = convert_car_orientation(current_heading)
                left_heading_degrees = (heading_degrees + 90) % 360
                left_heading = convert_car_orientation(left_heading_degrees)
                forward = DIRECTION_TO_VECTOR[current_heading]
                next_tile = tuple(np.array(forward) + np.array([x_curr, y_curr]))
                # need to iterate through to the last node of the left turn
                while left_heading not in plant.supervisor.game.map.legal_orientations[next_tile]:
                    next_tile = tuple(np.array(forward) + np.array([next_tile[0], next_tile[1]]))
                # reference state is that last subgoal of the left turn
                reference_state = plant.hack_state(plant.state, x=next_tile[0], y=next_tile[1], heading=left_heading)
                # intersection gap depends on agent heading
                if current_heading in ['east', 'west']:
                    intersection_gap = next_intersection.height
                else:
                    intersection_gap = next_intersection.width
                # find the first agent in the bundle that the agent will turn into
                lead_agent = plant.find_lead_agent(reference_state, must_not_be_in_intersection=True, same_heading_required=False)
                if lead_agent:
                    width, _= plant.get_width_along_bundle()
                    reference_state_bundle_width = width + 1
                    # defines the number of tiles in the intersection the agent has to cross-over to complete left turn
                    num_residual_tiles = intersection_gap - reference_state_bundle_width
                    # whichever one is bigger (one of them should be zero depending on agent heading)
                    clearance = max(abs(lead_agent.state.x-reference_state.x), abs(lead_agent.state.y-reference_state.y))
                else:
                    clearance = np.inf
                return clearance > intersection_gap #TODO: find a better bound

            else: # going straight
                lead_agent_in_intersection = plant.find_lead_agent(plant.state, must_not_be_in_intersection=False, same_heading_required=False)
                if lead_agent_in_intersection:
                    # if there's an agent in the intersection with a
                    # different heading ahead, don't enter
                    if lead_agent_in_intersection.state.heading != plant.state.heading:
                        return False
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

class UnprotectedLeftTurnOracle(Oracle):
    def __init__(self):
        super(UnprotectedLeftTurnOracle, self).__init__(name='unprotected_left_turn')

    def get_conservative_gap(self, lead_agent, N):
        """
        compute how much gap is needed for a continuous left turn for N
        time steps assuming maximum acceleration at each time step
        """
        gap = 0
        v_init = lead_agent.state.v
        for idx in range(N):
            v_init = min(lead_agent.v_max, v_init + lead_agent.a_max)
            gap += v_init
        return gap+1

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
                # idx indicates the lane number on the opposing traffic bundle
                # (bigger means closer to left-most lane)
                relative_tiles = [(idx, relative_occupancy[idx]) for idx in range(len(relative_occupancy))]
                relative_tiles.reverse()
                current_intersection = game.map.tile_to_intersection_map[current_directed_tile[0]]
                # get traffic light
                traffic_light = game.map.intersection_to_traffic_light_map[current_intersection]
                light_color = traffic_light.check_directed_light_in_N_turns(plant.state.heading, 0)
                if light_color == 'red':
                    # find agents in intersection to check if collision might occur during red light
                    for N, occupancy_tile in enumerate(relative_tiles):
                        abs_x, abs_y = opposing_bundle.relative_coordinates_to_tile(occupancy_tile)
                        fake_state = hack_car_state(plant.state, x=abs_x, y=abs_y, heading=plant.state.heading)
                        lead_agent = plant.find_lead_agent(fake_state, same_heading_required=False)
                        if lead_agent is not None:
                            # if agent is in intersection
                            if len(game.map.legal_orientations[(lead_agent.state.x, lead_agent.state.y)]) > 1:
                                # get other agent intention
                                if lead_agent.turn_signal == 'left':
                                    pass
                                else:
                                    # check collision, make conservative and assume other agent bundle hasn't gone
                                    gap = max(abs(abs_x-lead_agent.state.x), abs(abs_y-lead_agent.state.y))
                                    gap_requirement = self.get_conservative_gap(lead_agent, N+1)
                                    if gap > gap_requirement:
                                        pass
                                    else:
                                        return False
                            # other agent is not in intersection, don't need to check
                            else:
                                pass
                        else:
                            pass
                    return True
                else:
                    #print("checking for gap")
                    opposing_bundle, relative_occupancy = game.map.left_turn_to_opposing_traffic_bundles[current_directed_tile]
                    # idx indicates the lane number on the opposing traffic bundle
                    # (bigger means closer to left-most lane)
                    relative_tiles = [(idx, relative_occupancy[idx]) for idx in range(len(relative_occupancy))]
                    relative_tiles.reverse() # sort so temporal order is achieved
                    gaps = []
                    #fake_heading = opposing_bundle.direction
                    left_turn_gap_arr = []
                    for N, occupancy_tile in enumerate(relative_tiles):
                        abs_x, abs_y = opposing_bundle.relative_coordinates_to_tile(occupancy_tile)
                        fake_state = hack_car_state(plant.state, x=abs_x, y=abs_y, heading=plant.state.heading)
                        lead_agent = plant.find_lead_agent(fake_state, same_heading_required=False)

                        if lead_agent is None:
                            #print('no lead agent')
                            pass
                        else:
                            gap = max(abs(abs_x-lead_agent.state.x), abs(abs_y-lead_agent.state.y))
                            # TODO: complete gap conditions
                            gap_requirement = self.get_conservative_gap(lead_agent, N+1)
                            left_turn_gap_arr.append((gap_requirement, gap, N+1, lead_agent.state.x, lead_agent.state.y, abs_x, abs_y))
                            if gap > gap_requirement:
                                pass
                            else:
                                plant.left_turn_gap_arr = left_turn_gap_arr
                                return False
                    plant.left_turn_gap_arr = left_turn_gap_arr
                    return True
        else: # if the agent is not trying to perform a left turn
            return True

class NoDeadlockOracle(Oracle):
    def __init__(self):
        super(NoDeadlockOracle, self).__init__(name="no_deadlock")

    def evaluate(self, ctrl_action, plant, game, save_debug=False):
        ''' action evaluates to True only if action will maintain one-or-two gaps
        depending on whether agent is at an intersection and direction agent wants to turn, i.e.
        internal or external loop more specific version of the traffic clearance intersection
        currently only works for 4x4 intersections'''
        # want to record control action, case, clearance computed

        #no_deadlock_oracle_info_sv = od()

        # count agents in the intersection ahead 
        def count_agents_ahead(gap):
            cnt = 0
            # check for number of agents ahead in intersection
            forward = DIRECTION_TO_VECTOR[plant.state.heading]
            curr_st = np.array([plant.state.x, plant.state.y])
            for i in range(1, gap):
                next_tile_tuple = tuple(curr_st + i*np.array(forward))
                # if there is an agent there, then count it
                if next_tile_tuple in game.occupancy_dict:
                    cnt = cnt + 1
            return cnt
        
        # find the lead agent in the lane agent will turn left into and number of 
        # agents in intersection that are also making left turn
        def find_left_turn_lead_agent(plant, state=None):
            if state == None: 
                state = plant.state

            agent_cnt = 0
            x_curr, y_curr = state.x, state.y
            current_heading = state.heading
            heading_degrees = convert_car_orientation(current_heading)
            left_heading_degrees = (heading_degrees + 90) % 360
            left_heading = convert_car_orientation(left_heading_degrees)
            forward = DIRECTION_TO_VECTOR[current_heading]
            next_tile = tuple(np.array(forward) + np.array([x_curr, y_curr]))
            # need to iterate through to the last node of the left turn
            while left_heading not in plant.supervisor.game.map.legal_orientations[next_tile]:
                # check whether the agent is in the next tile and if it is, add it to the count
                agent_in_tile = next_tile in plant.supervisor.game.occupancy_dict
                if agent_in_tile:
                    agent_cnt = agent_cnt + 1
                
                next_tile = tuple(np.array(forward) + np.array([next_tile[0], next_tile[1]]))
            # reference state is that last subgoal of the left turn
            reference_state = plant.hack_state(state, x=next_tile[0], y=next_tile[1], heading=left_heading)
            lead_agent = plant.find_lead_agent(reference_state, must_not_be_in_intersection=True, same_heading_required=False)
            return lead_agent, agent_cnt, reference_state
        
        # TODO: need to generalize this TO TILES IN AGENT BUBBLE OR FOV
        # returns the appropriate tiles
        def get_rel_adjacent_tiles(plant):
            rel_tiles = od()
            # TODO: generalize this so field-of-view depends on agent bubble!!!!
            # default heading is north
            rel_tiles['left'] = [[-2,-1], [-1, -1], [0, -1], [1, -1], [2, -1]]
            rel_tiles['right'] = [[-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1]]
            return rel_tiles
        
        # returns true if the first tile is open and either second or third tile are unoccupied
        def check_clearance_gap_for_lane_change(plant, ctrl_action):
            # rotate the vector from 
            def relativize_displacement(plant, abs_dis): 
                rel_dis = []
                offset = np.array([plant.state.x, plant.state.y])
                for displacement in abs_dis:
                    heading = convert_car_orientation(plant.state.heading) * np.pi/180
                    arr=(offset + rotate_vector(tuple(displacement), heading))
                    rel_dis.append((arr[0], arr[1]))
                return rel_dis
            
            rel_tiles = get_rel_adjacent_tiles(plant)
            
            # convert the tiles to absolute coordinates 
            if ctrl_action['steer'] == 'left-lane':
                tiles = relativize_displacement(plant, rel_tiles['left'])
            elif ctrl_action['steer'] == 'right-lane':
                tiles = relativize_displacement(plant, rel_tiles['right'])
            else:
                print("ERROR, not a lane change action")
            
            # loop through all tiles and count how many gaps 
            cnt = 0 
            for tile in tiles: 
                if tile not in plant.supervisor.game.occupancy_dict: 
                    cnt = cnt + 1
            
            # check if agent subgoal is a left-turn tile, then look for clearance gaps
            # in lane where agent will left turn into 
            current_subgoal = plant.supervisor.subgoals[0]
            if current_subgoal in plant.supervisor.game.map.all_left_turns:
                # if at least one gap is available, then add to count by one
                # from state where you would start left-turn...
                st_f = plant.query_car_occupancy(ctrl_action)[-1]
                lead_agent_left, agent_ahead_cnt, ref_state = find_left_turn_lead_agent(plant, state=None)
                gap_total = max(abs(lead_agent_left.state.x-ref_state.x), abs(lead_agent_left.state.y-ref_state.y))
                clearance_gaps = gap_total-agent_ahead_cnt
                if clearance_gaps >= 1: 
                    cnt = cnt+1

            # need clearance of two 
            return cnt>=2, cnt

        # printing to debug what's happening
        #print(plant.state)
        #print(ctrl_action)

        # collect all relevant state info and compute intersection gap
        current_state = plant.state.x, plant.state.y
        x_curr, y_curr = current_state
        next_state = plant.query_next_state(ctrl_action)
        x_next, y_next = next_state.x, next_state.y
        bp_state = plant.query_occupancy(plant.get_backup_plan_ctrl(), state=next_state)[-1]
        x_next_bp, y_next_bp = bp_state.x, bp_state.y
        current_heading = plant.state.heading
        ctrl_action_str = ctrl_action['steer'] + ' ' + str(ctrl_action['acceleration'])

        # When agent is NOT CROSSING INTO INTERSECTION
        try:
            currently_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_curr,y_curr))
            will_be_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_next,y_next))
            bp_will_be_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_next_bp,y_next_bp))
        except:
            return True

        if currently_in_intersection or not will_be_in_intersection or not bp_will_be_in_intersection:
            #====================================================================================================#
            # CASE E: when agents want to do lane-change AND not go through an intersection
            # should check whether there are 2 gaps of clearance in loop agent wants to change lanes into
            #====================================================================================================#
            if ((ctrl_action['steer'] == 'left-lane') or (ctrl_action['steer'] == 'right-lane')):
                gap_is_enough, gap_cnt = check_clearance_gap_for_lane_change(plant, ctrl_action)
                if save_debug: 
                    plant.no_deadlock_sv[ctrl_action] = ['E', gap_cnt, gap_is_enough]
                return gap_is_enough

            #====================================================================================================#
            # CASE F: when agents not doing lane change AND not go through an intersection
            # oracle should be satisfied vacuously in this case
            #====================================================================================================#
            else: 
                if save_debug:
                    plant.no_deadlock_sv[ctrl_action] = ['F', None, None]
                return True

        # When the action will take the agent into the intersection
        else:
            if (ctrl_action['steer'] == 'right-lane') or (ctrl_action['steer'] == 'left-lane'):
                return True
            
            next_intersection = game.map.tile_to_intersection_map[(x_next, y_next)]

            # get traffic light information
            current_subgoal = plant.supervisor.subgoals[0]
            #print(current_subgoal)
            traffic_light = game.map.intersection_to_traffic_light_map[next_intersection]
            light_color = traffic_light.check_directed_light_in_N_turns(plant.state.heading, 0)
            
            # TODO: need to generalize to more types of intersections
            # specific left turn check, need to generalize 
            left_turn_chk = plant.supervisor.game.map.tile_is_in_intersection((current_subgoal[0][0], current_subgoal[0][1]))


            if left_turn_chk: 
                #====================================================================================================#
                # CASE A: when agents wants to do left turn 
                #====================================================================================================#
                if light_color == 'red':
                    if save_debug:
                        plant.no_deadlock_sv[ctrl_action] = ['A', 'red']
                    return True
                else:
                    # need to check 
                    # find lead agent in road that agent will left-turn into 
                    lead_agent_left, agent_ahead_cnt, ref_state = find_left_turn_lead_agent(plant)

                    # compute the clearance gap
                    if lead_agent_left:
                        # compute the distance between agents 
                        gap_total = max(abs(lead_agent_left.state.x-ref_state.x), abs(lead_agent_left.state.y-ref_state.y))
                        clearance_gaps = gap_total-agent_ahead_cnt
                        if clearance_gaps > 0: 
                            if save_debug: 
                                plant.no_deadlock_sv[ctrl_action] = ['A', clearance_gaps, True]
                            return True
                        else:
                            if save_debug: 
                                plant.no_deadlock_sv[ctrl_action] = ['A', clearance_gaps, False]
                            return False
                    # lead agent is not found, lots of spaces so OK
                    else:
                        if save_debug: 
                            plant.no_deadlock_sv[ctrl_action] = ['A', 10000, True]
                        return True 
                    

            # if action is to go straight in intersection
            elif ((not left_turn_chk) and ctrl_action['steer'] == 'straight'):
                #====================================================================================================#
                # CASE B: going straight ahead into intersection (not in right-most lane)
                # CASE C: going straight ahead into intersection (in right-most lane), just add into the agent count
                #====================================================================================================#
                directed_tile = (current_state, plant.state.heading)
                in_rightmost_lane = plant.supervisor.game.map.check_if_right_turn_tile(directed_tile)
                #if not in_rightmost_lane:  
                    # if red light, satisfied vacuously (shouldn't be going)???
                    # if green light, need at least two gaps of clearance
                    # find lead agent
                if light_color == 'red':
                    if save_debug: 
                        plant.no_deadlock_sv[ctrl_action] = ['B/C', 'red']
                    return True
                else: 
                    lead_agent_forward = plant.find_lead_agent(plant.state, must_not_be_in_intersection=False, same_heading_required=False)

                    # count how many agents between lead agent and current agent when there is a lead agent
                    if lead_agent_forward is not None: 
                        gap_between_agents = max(abs(lead_agent_forward.state.x-plant.state.x), abs(lead_agent_forward.state.y-plant.state.y))
                        agents_ahead = count_agents_ahead(gap_between_agents)
                        # compute how many agents in right-turn lane can fill in gaps
                        if in_rightmost_lane: 
                            # TODO: ADD IN COUNT, GENERALIZE LATER
                            #pass
                            agents_ahead = agents_ahead + 1
                            # compute gaps of clearance

                        clearance_gaps = gap_between_agents-agents_ahead
                        if clearance_gaps > 1: 
                            if save_debug: 
                                plant.no_deadlock_sv[ctrl_action] = ['B/C', clearance_gaps, True]
                            return True
                        else:
                            if save_debug: 
                                plant.no_deadlock_sv[ctrl_action] = ['B/C', clearance_gaps, False]
                            return False

                        # TODO: if in the right-most lane, then need to figure out how many agents might turn right into
                        # its lane
                        #no_deadlock_sv[ctrl_action] = ['C', clearance_gaps, False]
                        #return False
                    # no lead agent
                    else:
                        if save_debug: 
                            plant.no_deadlock_sv = ['C', 'nolead']
                        return True

            #====================================================================================================#
            # CASE D: Making a right-turn into an intersection
            #====================================================================================================#
            # if action is to turn right across intersection
            elif ctrl_action['steer'] == 'right-turn':
                right_turn_state = plant.query_occupancy(ctrl_action)[-1]
                right_turn_tile = (right_turn_state.x, right_turn_state.y)
                # if agent is in tile, return false
                if right_turn_tile in plant.supervisor.game.occupancy_dict:
                    if save_debug:
                        plant.no_deadlock_sv[ctrl_action] = ['D', 'agent in tile']
                    return False
                # if agent is not in tile
                else:
                    # when light color is red, other cars might go first
                    if light_color == 'red':
                        if save_debug: 
                            plant.no_deadlock_sv[ctrl_action] = ['D', False]
                        # TODO: WRITE FUNCTION TO CHECK WHETHER CAR HAS PRECEDENCE
                        #agent_ahead = check_whether_car_has_precedence()
                        # there is a car that can go ahead of agent
                        #if agent_ahead: 
                        #    return False
                        #else:
                        #    return True
                        return False
                    # otherwise, no other cars will have precedence over right turn agent
                    else:
                        if save_debug: 
                            plant.no_deadlock_sv[ctrl_action] = ['D', True]
                        return True

            # not left turn, straight across intersection OR right turn
            else: 
                print(ctrl_action)
                print(plant.state)
                print("ERROR!!")
        pass


class ImprovementBundleProgressOracleOne(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(ImprovementBundleProgressOracleOne, self).__init__(name="improve_progress_one")
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

                # 

                # check if strictly improving longitudinal/lateral distance
                latt_improves = (dlatt_next < dlatt_curr)
                long_improves = (dlong_next < dlong_curr)

                dlatt = dlatt_curr-dlatt_next
                dlong = dlong_curr-dlong_next
                improves = (dlatt + dlong) > 0

                # only need to check reachability for braking backup plan because this implies reachability for current action
                return improves
            else:
                return False

class ImprovementBundleProgressOracleTwo(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(ImprovementBundleProgressOracleTwo, self).__init__(name="improve_progress_two")
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

                # 

                # check if strictly improving longitudinal/lateral distance
                latt_improves = dlatt_next < dlatt_curr
                long_improves = dlong_next < dlong_curr
                improves = latt_improves or long_improves

                dlatt = dlatt_curr-dlatt_next
                dlong = dlong_curr-dlong_next
                improves = (dlatt + dlong) > 1

                # only need to check reachability for braking backup plan because this implies reachability for current action
                return improves
            else:
                return False

class ImprovementBundleProgressOracleThree(Oracle):
    # requires a supervisor controller
    def __init__(self):
        super(ImprovementBundleProgressOracleThree, self).__init__(name="improve_progress_three")
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

                # 

                # check if strictly improving longitudinal/lateral distance
                latt_improves = dlatt_next < dlatt_curr
                long_improves = dlong_next < dlong_curr
                improves = latt_improves or long_improves

                dlatt = dlatt_curr-dlatt_next
                dlong = dlong_curr-dlong_next
                improves = (dlatt + dlong) > 2

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
        #print(plant.supervisor.game.map.special_goal_tiles)
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
        
        # When agent is NOT CROSSING INTO INTERSECTION
        currently_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((plant.state.x,plant.state.y))
        next_state = plant.query_next_state(ctrl_action)
        x_next, y_next = next_state.x, next_state.y
        will_be_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_next,y_next))
            #bp_state = plant.query_occupancy(plant.get_backup_plan_ctrl(), state=next_state)[-1]
            #x_next_bp, y_next_bp = bp_state.x, bp_state.y
            #bp_will_be_in_intersection = plant.supervisor.game.map.tile_is_in_intersection((x_next_bp,y_next_bp))
        #except:
        #   return True

        # if agent is currently in intersection or action will take agent into intersection
        if currently_in_intersection or will_be_in_intersection:
            ego_tile = plant.state.x, plant.state.y
            ego_heading = plant.state.heading
            bundles = game.map.tile_to_bundle_map[plant.state.x,plant.state.y]
            # check if ego_heading aligns with any of bundles at the tile
            chk_heading_match = check_heading_matches_bundle_heading(bundles, ego_heading)

            # TODO: change
            if not chk_heading_match or (ego_tile, ego_heading) in game.map.special_goal_tiles:
                return True
            # else don't make a lane change in an intersection
            else:
                if ctrl_action['steer'] == 'left-lane' or ctrl_action['steer'] == 'right-lane':
                    return False
                else:
                    return True

        elif game.map.legal_orientations[(plant.state.x, plant.state.y)] is None:
            return True
        elif len(game.map.legal_orientations[(plant.state.x, plant.state.y)]) <= 1:
            return True
        else:
            print('unknown case')
            return False

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

class BackupPlanSafetyOracle(Oracle):
    def __init__(self):
        super(BackupPlanSafetyOracle, self).__init__(name='backup_plan_safety')
    def evaluate(self, ctrl_action, plant, game):
        # check if collision occurs by taking that action
        collision_chk = plant.check_collision_in_bubble(ctrl_action)
        if collision_chk:
            #print('collison occurred, returning false')
            return False
        else:
            # check if makes collision with agent during action
            next_state = plant.query_occupancy(ctrl_action)[-1]

            x, y, heading, v = next_state.x, next_state.y, next_state.heading, next_state.v
            lead_agent = plant.find_lead_agent(state=next_state, inside_bubble=True, same_heading_required=False)

            if lead_agent:
                x_a, y_a, v_a = lead_agent.state.x, lead_agent.state.y, lead_agent.state.v
                gap_curr = math.sqrt((x_a-x)**2 + (y_a-y)**2)
                # record lead agent
                plant.lead_agent = (lead_agent.state.__tuple__(), lead_agent.get_id(), lead_agent.agent_color, gap_curr)
                # record computed gap
                plant.gap_curr = gap_curr
                return compute_gap_req_fast(lead_agent.a_min, v_a, plant.a_min, v) <= gap_curr
            else:
                return True

def run(runnable_set):
    for runnable in runnable_set:
        runnable.run()


        '''def extract_right_turn_tiles(intersection, heading, cnt):
            #returns a list of tiles that corresponds to the right-turn tiles at
            #a specified 'intersection' that are facing the specified 'heading'
            #the count determines the length of the list of tiles returned
            
            def get_intersection_corners(intersection):
                x_lo, x_hi = min(list(zip(*intersection.grid))[0]), max(list(zip(*intersection.grid))[0])
                y_lo, y_hi = min(list(zip(*intersection.grid))[1]), max(list(zip(*intersection.grid))[1])
                return x_lo, x_hi, y_lo, y_hi

            x_min, x_max, y_min, y_max = get_intersection_corners(intersection)
            right_turn_tiles = []

            # when heading of agents is: north
            if heading == 'north': 
                tile_corner = max_x, max_y
                for i in range(1, cnt):
                    right_turn_tiles.append(tile_corner[0] + i, tile_corner[1])
            # east
            elif heading == 'east':
                tile_corner = max_x, min_y
                for i in range(1, cnt):
                    right_turn_tiles.append(tile_corner[0], tile_corner[1]-i)
                
            # south 
            elif heading == 'south':
                tile_corner = min_x, min_y
                for i in range(1, cnt):
                    right_turn_tiles.append(tile_corner[0] - i, tile_corner[1])

            # west
            else: 
                tile_corner = min_x, max_y
                for i in range(1, cnt):
                    right_turn_tiles.append(tile_corner[0], tile_corner[1]+i)

            return right_turn_tiles'''
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
    dx = int(np.sum([vel+(k+1)*a_min for k in range(dt)]))
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
            traffic_light = game.map.tile_to_traffic_light_map[(plant.state.x, plant.state.y)]
            light_is_red = self.check_if_light_red_in_N_turns(traffic_light, plant.state.heading, 0) # N=0
            if light_is_red:
                # check if right turn is valid
                return plant.check_right_turn_is_clear(ctrl_action)
            else:
                return action_not_running_a_red_light and backup_plant_will_still_be_ok
        else:
            return action_not_running_a_red_light and backup_plant_will_still_be_ok

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
        return gap

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
                return True
            # else don't make a lane change in an intersection
            else:
                if ctrl_action['steer'] == 'left-lane' or ctrl_action['steer'] == 'right-lane':
                    return False
                else:
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

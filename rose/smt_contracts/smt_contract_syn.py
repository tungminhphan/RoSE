"""
Tung Phan
"""
from eventlet.timeout import Timeout
import timeout_decorator
import scipy.interpolate
import subprocess
from ipdb import set_trace as st
from pysmt.shortcuts import (Symbol, LE, GE, Int, And, Or, Equals,
        NotEquals, Plus, Minus, Solver, ExactlyOne, Iff, Ite, Not,
        AtMostOne, Max)
from pysmt.typing import INT, BOOL
from collections import OrderedDict as od
import numpy as np
import random
import matplotlib.pyplot as plt

def Abs(x):
    return Max(x, -x)

class SMTGridder:
    def __init__(self, name, init_state, goal_state, color='r'):
        self.init_state = init_state
        self.goal_state = goal_state
        self.name = name
        self.state_variables = od.fromkeys(['x', 'y'], INT)
        self.color = color

    def get_constraints(self, T):
        for state_variable in self.state_variables:
            setattr(self, state_variable, [None] * T)
            for t in range(T):
                var = getattr(self, state_variable)
                var[t] = Symbol(self.name+'_'+state_variable+str(t), self.state_variables[state_variable])

        init_state_constraint = And(Equals(self.x[0], Int(self.init_state[0])),
                                    Equals(self.y[0], Int(self.init_state[1])))

        all_dynamic_constraints = []
        # TODO: may need to try a different encoding to avoid using
        # max/abs?
        for t in range(T-1):
            all_dynamic_constraints.append(LE(Plus(Abs(Minus(self.x[t+1], self.x[t])),
                                                   Abs(Minus(self.y[t+1], self.y[t]))),
                                                   Int(1)))
        final_state_constraint = And(Equals(self.x[T-1], Int(self.goal_state[0])),
                                     Equals(self.y[T-1], Int(self.goal_state[1])))
        return And([init_state_constraint]+all_dynamic_constraints+[final_state_constraint])

    def get_step_counter_variable(self, T):
        return Symbol(self.name+'_'+'counter'+'_'+str(T), INT)

    def get_state_variables_at(self, t):
        x = Symbol(self.name + '_' + 'x' + str(t),
                    self.state_variables['x'])
        y = Symbol(self.name + '_' + 'y' + str(t),
                    self.state_variables['y'])
        return x, y

    def get_step_count_constraint(self, T):
        moves = []
        counter_var = self.get_step_counter_variable(T)
        for t in range(T-1):
            x, y = self.get_state_variables_at(t)
            x_next, y_next = self.get_state_variables_at(t+1)
            moves.append(Ite(And(Equals(x, x_next), Equals(y, y_next)), Int(0), Int(1)))
        return Equals(counter_var, Plus(moves))

    def get_solved_states_at(self, solver, t):
        states = od()
        for state_variable  in self.state_variables:
            var = Symbol(self.name + '_' + state_variable + str(t), self.state_variables[state_variable])
            states[state_variable] = solver.get_value(var)
        return states

    def print_solved_values(self, solver, T):
        t = 0
        while t<T:
            printout = []
            for state_variable  in self.state_variables:
                var = Symbol(self.name + '_' + state_variable + str(t),
                        self.state_variables[state_variable])
                printout.append("%s = %s" %(var, solver.get_value(var)))
            print(printout)
            t += 1

def get_obstacle_collision_constraints(agent, obstacles, T):
    constraints = []
    for t in range(T):
        for obstacle in obstacles:
            ox, oy = obstacle
            x, y = agent.get_state_variables_at(t)
            constraints.append(AtMostOne(Equals(x, Int(ox)), Equals(y, Int(oy))))
    return constraints

def get_pairwise_collision_constraints(agent1, agent2, T):
    constraints = []
    for t in range(T):
        equals = []
        x1, y1 = agent1.get_state_variables_at(t)
        x2, y2 = agent2.get_state_variables_at(t)
        constraints.append(Or(NotEquals(x1, x2),
                              NotEquals(y1, y2)))
        if t < T:
            exchange = []
            x1n, y1n = agent1.get_state_variables_at(t+1)
            x2n, y2n = agent2.get_state_variables_at(t+1)
            exchange.append(NotEquals(x1, x2n))
            exchange.append(NotEquals(y1, y2n))
            exchange.append(NotEquals(x2, x1n))
            exchange.append(NotEquals(y2, y1n))
            constraints.append(Or(exchange))
    return And(constraints)

class SMTGame:
    def __init__(self, agents, T, extent=[], obstacles=[]):
        self.agents = agents
        self.T = T
        self.extent = extent
        self.obstacles = obstacles

    def get_agent_extent_constraints(self, agent):
        extent_constraints = []
        if self.extent:
            xmin = Int(self.extent[0])
            xmax = Int(self.extent[1])
            ymin = Int(self.extent[2])
            ymax = Int(self.extent[3])
            for t in range(T):
                xt, yt = agent.get_state_variables_at(t)
                extent_constraints.append(And(LE(xmin, xt),
                                              LE(xt, xmax),
                                              LE(ymin, yt),
                                              LE(yt, ymax)))
        return extent_constraints

    def get_constraints(self, counter_constraint=None):

        # individual agent constrants
        agent_constraint_list = []
        for agent in self.agents:
            # get agent dynamical constraints
            agent_constraint_list.append(agent.get_constraints(self.T))
            # get game extent constraints for agent
            agent_constraint_list = agent_constraint_list + self.get_agent_extent_constraints(agent)

        # collision constraints
        collision_constraints = []
        for agent_idx in range(len(self.agents)-1):
            for next_agent_idx in range(agent_idx+1, len(self.agents)):
                agent = self.agents[agent_idx]
                next_agent = self.agents[next_agent_idx]
                collision_constraints.append(get_pairwise_collision_constraints(agent,
                                             next_agent, self.T))

        # obstacle constraints
        obstacle_constraints = []
        for agent in self.agents:
             obstacle_constraints = obstacle_constraints + get_obstacle_collision_constraints(agent, obstacles, T)

        # counter constraint
        if counter_constraint:
            c_constraints = []
            c_vars = []
            for agent in self.agents:
                c_constraints.append(agent.get_step_count_constraint(T))
                c_vars.append(agent.get_step_counter_variable(T))
            counter_sat = [LE(Plus(c_vars), Int(counter_constraint))]
        agent_constraint_list = agent_constraint_list + collision_constraints + obstacle_constraints + c_constraints + counter_sat
        return And(agent_constraint_list)

def create_random_gridders(N, extent):
    x_extent = extent[0:2]
    y_extent = extent[2:]
    gridders = []
    unused_locations = [(x,y) for x in range(x_extent[0], x_extent[1])
            for y in range(y_extent[0], y_extent[1])]
    for i in range(N):
        # sample start position
        start_pos, end_pos = random.sample(unused_locations, 2)
        unused_locations.remove(start_pos)
        unused_locations.remove(end_pos)
        gridder = SMTGridder(init_state=start_pos, goal_state=end_pos,
                             name='gridder' + str(i), color='C' + str(i))
        gridders.append(gridder)
    return gridders

def create_random_obstacles(N, extent, agents):
    x_extent = extent[0:2]
    y_extent = extent[2:]
    unused_locations = [(x,y) for x in range(x_extent[0], x_extent[1])
            for y in range(y_extent[0], y_extent[1])]
    for agent in agents:
        init_state = agent.init_state
        goal_state = agent.goal_state
        unused_locations.remove(init_state)
        unused_locations.remove(goal_state)
    obstacles = random.sample(unused_locations, N)
    return obstacles


def make_partition_game():
    N = 6
    all_nodes = [(i,j) for i in range(N) for j in range(N)]
    node_to_index_dict = {node: node_idx for node_idx, node in enumerate(all_nodes)}

    io_map = od()
    io_map[(0,0)] = [(2,N-1), (3,N-1)] # input to outputs
    io_map[(4,0)] = [(N-1,N-1)]

    # for each of these, we will create a different color

    # build variable dictionary for whole grid
    vars_dict = od()
    for inp_idx, inp in enumerate(io_map.keys()):
        input_var_name = 'in'+str(inp_idx)
        output_var_name = 'out'+str(inp_idx)
        vars_dict[inp] = [None] * len(all_nodes)
        for out in io_map[inp]:
            vars_dict[out] = [None] * len(all_nodes)
        for node_idx, node in enumerate(all_nodes):
            var_name = input_var_name+'_'+str(node_idx)
            vars_dict[inp][node_idx] = Symbol(var_name)
            for out_idx, out in enumerate(io_map[inp]):
                var_name = output_var_name + '_' + str(out_idx) + '_' + str(node_idx) # out + in_idx + out_idx + node_idx
                vars_dict[out][node_idx] = Symbol(var_name)

    vars_io_map = od() # I/O map/variable (name) version
    vars_dict_with_var_key = od() # same as vars_dict but with variables as keys
    for inp in io_map:
        node_idx = node_to_index_dict[inp]
        inp_var = vars_dict[inp][node_idx]
        vars_io_map[inp_var] = []
        vars_dict_with_var_key[inp_var] = vars_dict[inp]
        for out in io_map[inp]:
            node_idx = node_to_index_dict[out]
            out_var = vars_dict[out][node_idx]
            vars_io_map[inp_var].append(out_var)
            vars_dict_with_var_key[out_var] = vars_dict[out]

    return vars_io_map, vars_dict_with_var_key

def add_game_constraints(io_map, vars_dict):
    constraints = []
    all_vars = [var for io_var in vars_dict for var in vars_dict[io_var]]
    init_vars = []
    for io_var in vars_dict:
        # input/output must be colored correctly
        constraints.append(io_var)
        init_vars.append(io_var)
        io_var_idx = vars_dict[io_var].index(io_var)
        # input/output must not be colored with anything else
        other_colors = [vars_dict[var][io_var_idx] for var in vars_dict if var != io_var]
        init_vars = init_vars + other_colors
        constraints.append(Not(Or(other_colors)))

    # each non-initial node must belong to a unique player
    players = dict()
    for inp in io_map:
        players[inp] = []
        for inp_var_idx, inp_var in enumerate(vars_dict[inp]):
            player_vars = []
            if inp_var not in init_vars:
                player_vars.append(inp_var)
                for out in io_map[inp]:
                    player_vars.append(vars_dict[out][inp_var_idx])
                players[inp].append(player_vars)
    N = len(players[inp]) # depends on loop before

    prod_set = [[And(players[player][node_idx]) for player in players] for
        node_idx in range(N)]

    # add unique player constraints
    for constr in prod_set:
        constraints.append(ExactlyOne(constr))

    # add connectivity constraints
    st()

    return constraints

def find_neighbors(node, node_list):
    return [(node[0]+i*j, node[1]+(i+1)%2*j) for i in [0, 1]
            for j in [-1, 1] if (node[0]+i*j,
            node[1]+(i+1)%2*j) in node_list]

infos = make_partition_game()
add_game_constraints(infos[0], infos[1])
st()

if __name__ == '__main__':
    random.seed(0)
    T = 6
    N_obstacles = 6
    N_gridders = 4
    step_count_max = 15
    map_length = 7
    x_extent = [2, map_length]
    y_extent = [2, map_length]
    extent = x_extent + y_extent
    # randomly create robots
    gridders = create_random_gridders(N=N_gridders, extent=extent)
    # randomly generate obstacles
    obstacles = create_random_obstacles(N=N_obstacles, extent=extent, agents=gridders)
    game = SMTGame(agents=gridders, T=T, extent=extent, obstacles=obstacles)
    constraints = game.get_constraints(counter_constraint=step_count_max)
    N_interp = 4
#    try:
#        timeout = Timeout(2, 'timed out!!')
#        while timeout:
#            pass
#    finally:
#        timeout.cancel()

    with Solver(name='z3') as solver:
        solver.add_assertion(constraints)
        # clear all figures in /figs/
        subprocess.run('rm ./figs/*.png', shell=True)
        fig = plt.figure()
        if solver.solve():
            print('Found a solution!')
            for t in range((game.T-1) * N_interp + 1):
                fig.clf()
                lower_t = t // N_interp
                higher_t = lower_t + 1
                for obstacle in game.obstacles:
                    ox, oy = obstacle
                    plt.plot(ox, oy, 'bs', markersize=10)
                for agent in game.agents:
                    prev_agent_states = agent.get_solved_states_at(solver, lower_t)
                    prev_x = int(str(prev_agent_states['x']))
                    prev_y = int(str(prev_agent_states['y']))

                    next_agent_states = agent.get_solved_states_at(solver, higher_t)
                    next_x = int(str(next_agent_states['x']))
                    next_y = int(str(next_agent_states['y']))

                    x_interp = scipy.interpolate.interp1d([lower_t, higher_t], [prev_x, next_x])
                    y_interp = scipy.interpolate.interp1d([lower_t, higher_t], [prev_y, next_y])
                    x = x_interp(t / N_interp)
                    y = y_interp(t / N_interp)
                    plt.plot(x, y, agent.color+'o', markersize=15,
                            alpha=1)
                    gx, gy = agent.goal_state
                    plt.plot(gx, gy, agent.color+'x', markersize=20)
                print(t)
                padding = 0.2
                plt.axis('scaled')
                plt.xlim(game.extent[0]-padding, game.extent[1]+padding)
                plt.ylim(game.extent[2]-padding, game.extent[3]+padding)
                fig.savefig('./figs/'+str(t).zfill(5)+'.png', dpi=fig.dpi)
        else:
            print('No solution found')
            fig = plt.figure()
            for obstacle in game.obstacles:
                ox, oy = obstacle
                plt.plot(ox, oy, 'bs', markersize=40)
            for agent in game.agents:
                agent_states = agent.init_state
                x = agent_states[0]
                y = agent_states[1]
                plt.plot(x, y, agent.color+'o', markersize=20)
                gx, gy = agent.goal_state
                plt.plot(gx, gy, agent.color+'x', markersize=20)
            plt.xlim(game.extent[0], game.extent[1])
            plt.ylim(game.extent[2], game.extent[3])
            fig.savefig('./figs/'+str(0).zfill(5)+'.png', dpi=fig.dpi)

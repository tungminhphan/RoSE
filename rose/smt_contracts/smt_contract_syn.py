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
        AtMostOne, Max, Implies)
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

class PartitionQuery:
    def __init__(self, node_list, io_map):
        self.node_list = node_list
        self.io_map = io_map
        self.vars_io_map, self.vars_dict_with_var_key = self.create_variable_maps()
        self.player_color_dict = self.get_player_colors()

    def create_variable_maps(self):
        # build variable dictionary for whole grid
        # each player controlling an input
        vars_dict = od()
        for inp_idx, inp in enumerate(io_map.keys()):
            input_var_name = 'player'+str(inp_idx)
            vars_dict[inp] = [None] * len(node_list)
            for node_idx, node in enumerate(node_list):
                var_name = input_var_name+'_'+str(node_idx)
                vars_dict[inp][node_idx] = Symbol(var_name, INT)

        vars_io_map = od() # I/O map/variable (name) version
        vars_dict_with_var_key = od()
        for inp in io_map:
            node_idx = node_list.index(inp)
            inp_var = vars_dict[inp][node_idx]
            vars_dict_with_var_key[inp_var] = vars_dict[inp]
            vars_io_map[inp_var] = []
            for out in io_map[inp]:
                node_idx = node_list.index(out)
                out_var = vars_dict[inp][node_idx]
                vars_io_map[inp_var].append(out_var)

        return vars_io_map, vars_dict_with_var_key

    def get_constraints(self):
        io_map = self.vars_io_map
        vars_dict = self.vars_dict_with_var_key
        constraints = []
        for inp_var in vars_dict:
            # inputs must have 0 distances
            constraints.append(Equals(inp_var, Int(0)))
            for other_var in [var for var in vars_dict[inp_var] if var != inp_var]:
                if other_var in io_map[inp_var]:
                    # if is an output var, must be reachable
                    constraints.append(GE(other_var, Int(1)))
                else:
                    # if is not i/o var
                    constraints.append(Or(GE(other_var, Int(1)), Equals(other_var, Int(-1))))

        # each non-initial node must belong to a unique player
        N = len(vars_dict[list(vars_dict.keys())[0]])
        for idx in range(N):
            player_choices = []
            for inp_var in vars_dict:
                var = vars_dict[inp_var][idx]
                var_chosen = NotEquals(var, Int(-1))
                player_choices.append(var_chosen)
            # each is chosen by only one player
            constraints.append(ExactlyOne(player_choices))

        # connectivity constraints
        for var in vars_dict:
            all_choices = vars_dict[var]
            for choice_idx, choice in enumerate(all_choices):
                if choice not in vars_dict:
                    choice_xy = self.node_list[choice_idx]
                    choice_neighbors_idx = [self.node_list.index(neighbor_xy) for neighbor_xy in find_neighbors(choice_xy, self.node_list)]
                    choice_neighbors = [all_choices[idx] for idx in choice_neighbors_idx]
                    connect = []
                    for neighbor in choice_neighbors:
                        connect.append(Implies(NotEquals(choice, Int(-1)),
                                               Equals(choice, Plus(neighbor, Int(1)))))
                    constraints.append(Or(connect))
        return And(constraints)

    def extract_solutions(self, solver):
        soln = dict()
        for node_idx, node in enumerate(self.node_list):
            soln[node] = []
            for in_var_idx, in_var in enumerate(self.vars_dict_with_var_key):
                var = self.vars_dict_with_var_key[in_var][node_idx]
                val = int(str(solver.get_value(var)))
                if val >= 0:
                    soln[node].append(list(self.io_map.keys())[in_var_idx])
        return soln

    def solve(self, solver='z3'):
        soln = None
        constraints = self.get_constraints()

        with Solver(name=solver) as solv:
            solv.add_assertion(constraints)
            if solv.solve():
                print('Found a solution!')
                soln = self.extract_solutions(solv)
            else:
                print('No solution found!')
        return soln

    def get_player_colors(self):
        color_dict = dict()
        for player_idx, player in enumerate(self.io_map):
            color_dict[player]= color='C' + str(player_idx)
        return color_dict

    def plot_solution(self):
        ans = self.solve()
        if not ans:
            print('Nothing to plot!')
        else:
            for node in ans:
                for player in ans[node]:
                    plt.plot(node[0], node[1], 'bs',
                            markersize=1000/len(self.node_list),
                            color=self.player_color_dict[player])
            plt.axis('scaled')
            plt.show()


def find_neighbors(node, node_list):
    return [(node[0]+i*j, node[1]+(i+1)%2*j) for i in [0, 1]
            for j in [-1, 1] if (node[0]+i*j,
            node[1]+(i+1)%2*j) in node_list]


N = 6
node_list = [(i,j) for i in range(N) for j in range(N)]
io_map = od()
io_map[(0,0)] = [(1,N-1), (2,N-1)] # input to outputs
io_map[(4,0)] = [(N-1,N-1)]
io_map[(2,0)] = [(4,N-1)]

query = PartitionQuery(node_list=node_list, io_map=io_map)
ans = query.plot_solution()
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

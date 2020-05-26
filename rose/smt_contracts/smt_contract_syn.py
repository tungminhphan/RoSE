"""
Tung Phan
"""
import subprocess
from ipdb import set_trace as st
from pysmt.shortcuts import (Symbol, LE, GE, Int, And, Or, Equals,
        NotEquals, Plus, Minus, Solver, ExactlyOne, Iff, Not,
        AtMostOne, Max)
from pysmt.typing import INT, BOOL
from collections import OrderedDict as od
import numpy as np
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
                var[t] = Symbol(self.name+'_'+state_variable+str(t), INT)

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

def get_pairwise_collision_constraints(agent1, agent2, T):
    constraints = []
    for t in range(T):
        equals = []
        if t < T:
            exchange = []
        for state_variable in agent1.state_variables:
            var1 = Symbol(agent1.name + '_' + state_variable + str(t),
                    agent1.state_variables[state_variable])
            var2 = Symbol(agent2.name + '_' + state_variable + str(t),
                    agent2.state_variables[state_variable])
            equals.append(Equals(var1, var2))
            if t < T:
                var1_next = Symbol(agent1.name + '_' + state_variable + str(t+1),
                        agent1.state_variables[state_variable])
                var2_next = Symbol(agent2.name + '_' + state_variable + str(t+1),
                        agent2.state_variables[state_variable])
                exchange.append(NotEquals(var1, var2_next))
                exchange.append(NotEquals(var2, var1_next))
        constraints.append(AtMostOne(equals))
        if t < T:
            constraints.append(Or(exchange))

    return And(constraints)

class SMTGame:
    def __init__(self, agents, T, extent=[]):
        self.agents = agents
        self.T = T
        self.extent = extent

    def get_agent_extent_constraints(self, agent):
        extent_constraints = []
        if self.extent:
            xmin = Int(self.extent[0])
            xmax = Int(self.extent[1])
            ymin = Int(self.extent[2])
            ymax = Int(self.extent[3])
            for t in range(T):
                xt = Symbol(agent.name + '_' + 'x' + str(t), agent.state_variables['x'])
                yt = Symbol(agent.name + '_' + 'y' + str(t), agent.state_variables['y'])
                extent_constraints.append(And(LE(xmin, xt),
                                              LE(xt, xmax),
                                              LE(ymin, yt),
                                              LE(yt, ymax)))
        return extent_constraints

    def get_constraints(self):
        agent_constraint_list = []
        for agent in self.agents:
            # get agent dynamical constraints
            agent_constraint_list.append(agent.get_constraints(self.T))
            # get game extent constraints for agent
            agent_constraint_list = agent_constraint_list + self.get_agent_extent_constraints(agent)

        collision_constraints = []
        for agent_idx in range(len(self.agents)-1):
            for next_agent_idx in range(agent_idx+1, len(self.agents)):
                agent = self.agents[agent_idx]
                next_agent = self.agents[next_agent_idx]
                collision_constraints.append(get_pairwise_collision_constraints(agent,
                                             next_agent, self.T))
        agent_constraint_list = agent_constraint_list + collision_constraints
        return And(agent_constraint_list)

def create_random_gridders(N, extent):
    x_extent = extent[0:2]
    y_extent = extent[2:]
    gridders = []
    used_positions = []
    for i in range(N):
        # sample start position
        x_s = np.random.choice(range(x_extent[0], x_extent[1]))
        y_s = np.random.choice(range(y_extent[0], y_extent[1]))
        while (x_s, y_s) in used_positions:
            x_s = np.random.choice(range(x_extent[0], x_extent[1]))
            y_s = np.random.choice(range(y_extent[0], y_extent[1]))
        start_pos = (int(x_s), int(y_s))
        used_positions.append(start_pos)
        # sample end position
        x_e = np.random.choice(range(x_extent[0], x_extent[1]))
        y_e = np.random.choice(range(y_extent[0], y_extent[1]))
        while (x_e, y_e) in used_positions:
            x_e = np.random.choice(range(x_extent[0], x_extent[1]))
            y_e = np.random.choice(range(y_extent[0], y_extent[1]))
        end_pos = (int(x_e), int(y_e))
        used_positions.append(end_pos)
        gridder = SMTGridder(init_state=start_pos, goal_state=end_pos,
                             name='gridder' + str(i), color='C' + str(i))
        gridders.append(gridder)
    return gridders


if __name__ == '__main__':
    np.random.seed(0)
    T = 9
    x_extent = [0, 5]
    y_extent = [0, 5]
    extent = x_extent + y_extent
    gridders = create_random_gridders(N=9, extent=extent)

    game = SMTGame(agents=gridders, T=T, extent=extent)
    constraints = game.get_constraints()
    with Solver(name='z3', logic='QF_LIA') as solver:
        solver.add_assertion(constraints)
        plt.axis('equal')
        # clear all figures in /figs/
        subprocess.run('rm ./figs/*.png', shell=True)
        if solver.solve():
            print('Found a solution!')
            for t in range(game.T):
                fig = plt.figure()
                for agent in game.agents:
                    dotted_lines = []
                    if t > 2:
                        # plot agent states two steps back
                        agent_states = agent.get_solved_states_at(solver, t-3)
                        x = int(str(agent_states['x']))
                        y = int(str(agent_states['y']))
                        dotted_lines.append([x,y])
                    if t > 1:
                        # plot agent states two steps back
                        agent_states = agent.get_solved_states_at(solver, t-2)
                        x = int(str(agent_states['x']))
                        y = int(str(agent_states['y']))
                        dotted_lines.append([x,y])
                    if t > 0:
                        # plot previous agent states
                        agent_states = agent.get_solved_states_at(solver, t-1)
                        x = int(str(agent_states['x']))
                        y = int(str(agent_states['y']))
                        dotted_lines.append([x,y])

                    agent_states = agent.get_solved_states_at(solver, t)
                    x = int(str(agent_states['x']))
                    y = int(str(agent_states['y']))
                    dotted_lines.append([x,y])
                    plt.plot(x, y, agent.color+'o', markersize=20,
                            alpha=1)
                    if len(dotted_lines) > 1:
                        dotted_lines = np.array(dotted_lines)
                        plt.plot(dotted_lines[:,0], dotted_lines[:,1],
                                agent.color+'--', alpha=0.5)
                    gx, gy = agent.goal_state
                    plt.plot(gx, gy, agent.color+'x', markersize=20)
                print(t)
                plt.xlim(game.extent[0]-1, game.extent[1]+1)
                plt.ylim(game.extent[2]-1, game.extent[3]+1)
                fig.savefig('./figs/'+str(t).zfill(5)+'.png', dpi=fig.dpi)
        else:
            print('No solution found')
            fig = plt.figure()
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

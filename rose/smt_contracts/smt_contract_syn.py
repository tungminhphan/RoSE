"""
Tung Phan
"""
from ipdb import set_trace as st
from pysmt.shortcuts import (Symbol, LE, GE, Int, And, Equals, Plus,
                             Minus, Solver, ExactlyOne, Iff, Not,
                             AtMostOne, Max)
from pysmt.typing import INT, BOOL
from collections import OrderedDict as od
import matplotlib.pyplot as plt

def Abs(x):
    return Max(x, -x)

class SMTGridder:
    def __init__(self, name, init_state, goal_state):
        self.init_state = init_state
        self.goal_state = goal_state
        self.name = name
        self.state_variables = od.fromkeys(['x', 'y'], INT)

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

def get_agent_pair_collision_constraint(agent1, agent2, T):
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
                exchange.append(Equals(var1, var2_next))
                exchange.append(Equals(var2, var1_next))
        constraints.append(AtMostOne(equals))
        if t < T:
            constraints.append(AtMostOne(exchange))

        if t != T-1: # if not last time step must disallow exchange as well
            for state_variable in agent1.state_variables:
                var1 = Symbol(agent1.name + '_' + state_variable +
                        str(t+1),
                        agent1.state_variables[state_variable])
                var2 = Symbol(agent2.name + '_' + state_variable +
                        str(t+1),
                        agent2.state_variables[state_variable])
                equals.append(Equals(var1, var2))

    return And(constraints)

class SMTGame:
    def __init__(self, agents, T):
        self.agents = agents
        self.T = T

    def get_constraints(self):
        agent_constraint_list = []
        for agent in self.agents:
            agent_constraint_list.append(agent.get_constraints(self.T))

        collision_constraints = []
        for agent_idx in range(len(self.agents)-1):
            agent = self.agents[agent_idx]
            next_agent = self.agents[agent_idx+1]
            collision_constraints.append(get_agent_pair_collision_constraint(agent,
                                            next_agent, self.T))
        agent_constraint_list = agent_constraint_list + collision_constraints
        return And(agent_constraint_list)

T = 6
gridders = []
gridder0 = SMTGridder(init_state=[0,0], goal_state=[1,3], name='robot0')
gridders.append(gridder0)
gridder1 = SMTGridder(init_state=[1,0], goal_state=[2,3], name='robot1')
gridders.append(gridder1)
gridder2 = SMTGridder(init_state=[2,0], goal_state=[1,2], name='robot2')
gridders.append(gridder2)
gridder3 = SMTGridder(init_state=[1,3], goal_state=[3,5], name='robot3')
gridders.append(gridder3)

game = SMTGame(agents=gridders, T=T, exent=[0,0,10, 10])
constraints = game.get_constraints()

with Solver(name='cvc4', logic="QF_LIA") as solver:
    solver.add_assertion(constraints)
    if solver.solve():
        for t in range(game.T):
            plt.axis('equal')
            plt.grid('on')
            for agent in game.agents:
                agent_states = agent.get_solved_states_at(solver, t)
                x = int(str(agent_states['x']))
                y = int(str(agent_states['y']))
                plt.plot(x, y, 'o', markersize=20)
            plt.show()
    else:
        print('No solution found')


"""
Tung Phan
"""

from ipdb import set_trace as st
from pysmt.shortcuts import (Symbol, LE, GE, Int, And, Equals, Plus,
                             Minus, Solver, ExactlyOne, Iff,
                             AtMostOne, Max)
from pysmt.typing import INT, BOOL

def Abs(x):
    return Max(x, -x)

class SMTGridder:
    def __init__(self, name, init_state, goal_state):
        self.init_state = init_state
        self.goal_state = goal_state
        self.name = name
        self.state_variables = ['x', 'y']
        self.state_variable_types = [INT, INT]

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

    def get_solved_values(self, solver, T):
        t = 0
        while t<T:
            printout = []
            for variable_idx, state_variable  in enumerate(self.state_variables):
                var = Symbol(self.name + '_' + state_variable + str(t),
                        self.state_variable_types[variable_idx])
                printout.append("%s = %s" %(var, solver.get_value(var)))
            print(printout)
            t += 1
        st()

T = 20
gridder0 = SMTGridder(init_state=[2,1], goal_state=[6,5], name='robot0')
constraints = gridder0.get_constraints(T)

with Solver(name='cvc4', logic="QF_LIA") as solver:
    solver.add_assertion(constraints)
    if solver.solve():
        gridder0.get_solved_values(solver, T)
    else:
        print("No solution found")

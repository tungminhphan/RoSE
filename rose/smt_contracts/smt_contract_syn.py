"""
Tung Phan
"""

from ipdb import set_trace as st
from pysmt.shortcuts import (Symbol, LE, GE, Int, And, Equals, Plus,
                             Minus, Solver, ExactlyOne, Iff,
                             AtMostOne, Max)
from pysmt.typing import INT, BOOL

def Abs(x):
    return Max(x, Minus(Int(0),-x))

class SMTGridder:
    def __init__(self, name, init_state, goal_state):
        self.init_state = init_state
        self.goal_state = goal_state
        self.name = name
        self.state_variables = ['x', 'y']

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

        return And([init_state_constraint]+all_dynamic_constraints+[final_state_constraint]

#        print(self.x)
#        print(self.y)
#        print(init_state_constraint)
#        print(all_dynamic_constraints)
#        print(final_state_constraint)

gridder0 = SMTGridder(init_state=[0,0], goal_state=[5,5], name='robot0')
constraints = gridder0.get_constraints(15)
print(constraints)
st()

x0 = Symbol('x0', INT)
y0 = Symbol('y0', INT)

init_state_constraint = And(Equals(x0, Int(0)),
                            Equals(y0, Int(0)))

up0 = Symbol('up0', BOOL)
down0 = Symbol('down0', BOOL)
left0 = Symbol('left0', BOOL)
right0 = Symbol('right0', BOOL)
stay0 = Symbol('stay0', BOOL)

act0 = [up0, down0, left0, right0, stay0]

x1 = Symbol('x1', INT)
y1 = Symbol('y1', INT)

up1 = Symbol('up1', BOOL)
down1 = Symbol('down1', BOOL)
left1 = Symbol('left1', BOOL)
right1 = Symbol('right1', BOOL)
stay1 = Symbol('stay1', BOOL)

act1 = [up1, down1, left1, right1, stay1]

x2 = Symbol('x2', INT)
y2 = Symbol('y2', INT)

state_variables = [x0, y0, x1, y1, x2, y2]
state_space_constraint = And(And(LE(Int(0), l),
                          GE(Int(2), l)) for l in state_variables)

one_action_constraint = And(ExactlyOne(act0),
                            ExactlyOne(act1))

################ define dynamics #####################
dynamics_constraint = And(left0.Implies(And(Equals(x1, Plus(x0, Int(-1))),
                                            Equals(y1, y0))),
                          right0.Implies(And(Equals(x1, Plus(x0, Int(1))),
                                             Equals(y1, y0))),
                          up0.Implies(And(Equals(y1, Plus(y0, Int(-1))),
                                          Equals(x1, x0))),
                          down0.Implies(And(Equals(y1, Plus(y0, Int(1))),
                                            Equals(x1, x0))),
                          stay0.Implies(And(Equals(x1, x0),
                                            Equals(y1, y0))),
                          left1.Implies(And(Equals(x2, Plus(x1, Int(-1))),
                                             Equals(y2, y1))),
                          right1.Implies(And(Equals(x2, Plus(x1, Int(1))),
                                             Equals(y2, y1))),
                          up1.Implies(And(Equals(y2, Plus(y1, Int(-1))),
                                          Equals(x2, x1))),
                          down1.Implies(And(Equals(y2, Plus(y1, Int(1))),
                                            Equals(x2, x1))),
                          stay1.Implies(And(Equals(x2, x1),
                                            Equals(y2, y1))))
######################################################
final_state_constraint = And(Equals(x2, Int(1)),
                            Equals(y2, Int(1)))

problem = And(init_state_constraint,
              state_space_constraint,
              one_action_constraint,
              final_state_constraint,
              dynamics_constraint)

print("Serialization of the formula:")
print(problem)

with Solver(name='cvc4', logic="QF_LIA") as solver:
    solver.add_assertion(problem)
    if solver.solve():
        for l in state_variables:
            print("%s = %s" %(l, solver.get_value(l)))
    else:
        print("No solution found")

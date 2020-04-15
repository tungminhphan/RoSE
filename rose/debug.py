import os
import _pickle as pickle
import random
import copy as cp

# make sure an agent is doing the right things


def print_one_agent_trace(filename):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    
    t_end = traces['t_end']

    # select an agent at random
    agent_ids = traces['agent_ids']
    agent_id = random.choice(agent_ids)
    agent_trace = traces[agent_id].copy()

    # get agent params then remove from trace
    agent_param = agent_trace['agent_param']
    del agent_trace['agent_param']    

    # sort time_steps

    # inspect the agent trace over time (how the agent is making it's decisions)
    for t in sorted(agent_trace.keys()): 

        # print out the time stamp
        trace_t = agent_trace[t]

        print("time step")
        print(t)

        # print the agent state
        print("AGENT IS LOCATED AT: ")
        print(trace_t['state'])

        # print the other agents in its bubble
        print("OTHER AGENTS IN BUBBLE ARE LOCATED AT: ")
        [print(ag) for ag in trace_t['agents_in_bubble']]

        # print the agents goal
        print("AGENT CGOAL IS:")
        print(trace_t['goals'])
        
        # print out oracle dict
        if t != list(sorted(agent_trace.keys()))[-1]:
            trace_nxt = agent_trace[t+1]

            # print out the oracle scores
            for ctrl, oracle_scores in trace_nxt['spec_struct_info'].items():
                print("control action")
                print(ctrl)
            
                for key, value in oracle_scores.items():
                    print(key, value)
        
            # print out the conflict requests it sent out
            print('sent requests to:')
            for agent in trace_nxt['sent_request']:
                print(agent)
            
            # print out the conflict requests received
            print('received requests from')
            for agent in trace_nxt['received_requests']:
                print(agent)

            # print token count
            print('agent token count after action is taken')
            print(trace_nxt['token_count'])

            # print out max braking flag
            print('max braking flag sent')
            print(trace_nxt['max_braking_not_enough'])

            # print control action taken
            print('control action taken')
            print(trace_nxt['action'])

            
        '''{'state':(agent.state.x, agent.state.y, agent.state.heading, agent.state.v), 'action': agent.ctrl_chosen, \
                'color':agent.agent_color, 'bubble':agent.get_bubble(), \
                  ,  \
                            }'''

        print()
        print()
        pass

# test out the debug file 
if __name__ == '__main__':
    #output_dir = os.getcwd()+'/imgs/'
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    traces_file = os.getcwd()+'/saved_traces/game.p'
    print_one_agent_trace(traces_file)
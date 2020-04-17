import os
import _pickle as pickle
import random
import copy as cp

# make sure an agent is doing the right things
def get_agent_id(filename, x, y, heading, t):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    agents = traces[t]['agents']
    # search through all agents at time t 
    for agent in agents:
        ag_x, ag_y, ag_theta, ag_v, ag_color, ag_bubble, ag_id = agent
        if ag_theta == heading: 
            print(ag_x, ag_y, ag_theta)
        if (x, y, heading) == (ag_x, ag_y, ag_theta):
            return ag_id
    
    print('agent not found')
    return None


def print_one_agent_trace(filename, outfile, x, y, heading, t):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    
    out_file = open(outfile,"w") 
    
    t_end = traces['t_end']

    # select an agent at random
    #agent_ids = traces['agent_ids']
    #agent_id = random.choice(agent_ids)
    agent_id = get_agent_id(filename, x, y, heading, t)
    agent_trace = traces[agent_id].copy()

    # get agent params then remove from trace
    agent_param = agent_trace['agent_param']
    del agent_trace['agent_param']    

    # sort time_steps

    # inspect the agent trace over time (how the agent is making it's decisions)
    for t in sorted(agent_trace.keys()): 

        # print out the time stamp
        trace_t = agent_trace[t]

        out_file.write("Time Step \n")
        out_file.write(str(t)+'\n')
        #print("time step")
        #print(t)

        # print the agent state
        out_file.write("AGENT IS LOCATED AT: \n")
        out_file.write(str(trace_t['state'])+'\n')
        #print("AGENT IS LOCATED AT: ")
        #print(trace_t['state'])

        # print the agent state
        out_file.write("AGENTS' BUBBLE INCLUDES: \n")
        out_file.write(str(trace_t['bubble'])+'\n')

        # print the other agents in its bubble
        out_file.write("OTHER AGENTS IN BUBBLE ARE LOCATED AT: \n")
        [out_file.write(str(ag)+'\n') for ag in trace_t['agents_in_bubble']]
        #print("OTHER AGENTS IN BUBBLE ARE LOCATED AT: ")
        #[print(ag) for ag in trace_t['agents_in_bubble']]

        # print the agents goal
        out_file.write("AGENT GOAL IS:\n")
        out_file.write(str((trace_t['goals']))+'\n')
        #print("AGENT GOAL IS:")
        #print(trace_t['goals'])
        
        # print out oracle dict
        if t != list(sorted(agent_trace.keys()))[-1]:
            trace_nxt = agent_trace[t+1]

            # print out the oracle scores
            for ctrl, oracle_scores in trace_nxt['spec_struct_info'].items():
                out_file.write("control action\n")
                out_file.write(str(ctrl)+'\n')
                #print("control action")
                #print(ctrl)
            
                for key, value in oracle_scores.items():
                    out_file.write(key + ' ' + str(value)+'\n')
                    #print(key, value)
        
            # print out the conflict requests it sent out
            out_file.write('sent requests to:\n')
            #print('sent requests to:')
            for agent in trace_nxt['sent_request']:
                out_file.write(str(agent)+'\n')
                #print(agent)
            
            # print out the conflict requests received
            out_file.write('received requests from\n')
            #print('received requests from')
            for agent in trace_nxt['received_requests']:
                out_file.write(str(agent)+'\n')
                #print(agent)

            # print token count
            out_file.write('agent token count after action is taken\n')
            out_file.write(str(trace_nxt['token_count'])+'\n')
            #print('agent token count after action is taken')
            #print(trace_nxt['token_count'])

            # print out max braking flag
            out_file.write('max braking flag sent\n')
            out_file.write(str(trace_nxt['max_braking_not_enough'])+'\n')
            #print('max braking flag sent')
            #print(trace_nxt['max_braking_not_enough'])

            # print control action taken
            out_file.write('control action taken\n')
            out_file.write(str(trace_nxt['action'])+'\n')
            #print('control action taken')
            #print(trace_nxt['action'])

            
        '''{'state':(agent.state.x, agent.state.y, agent.state.heading, agent.state.v), 'action': agent.ctrl_chosen, \
                'color':agent.agent_color, 'bubble':agent.get_bubble(), \
                  ,  \
                            }'''
        out_file.write('\n')
        out_file.write('\n')
        #print()
        #print()
        pass
    out_file.close()

# test out the debug file 
if __name__ == '__main__':
    #output_dir = os.getcwd()+'/imgs/'
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    traces_file = os.getcwd()+'/saved_traces/game.p'

    outfile = os.getcwd()+'/saved_traces/debug.txt'
    print_one_agent_trace(traces_file, outfile, 15, 37, 'north', 54)
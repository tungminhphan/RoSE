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
            print(ag_x, ag_y, ag_theta, ag_color)
        if (x, y, heading) == (ag_x, ag_y, ag_theta):
            return ag_id
    
    print('agent not found')
    return None

def print_one_agent_trace(filename, outfile, x, y, heading, t):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    
    out_file = open(outfile,"w") 
    
    t_end = traces['t_end']
    #print(t_end)

    # select an agent at random
    agent_id = get_agent_id(filename, x, y, heading, t)
    agent_trace = traces[agent_id].copy()

    # get agent params then remove from trace
    agent_param = agent_trace['agent_param']
    del agent_trace['agent_param']    

    #print(agent_trace.keys())
    # inspect the agent trace over time (how the agent is making it's decisions)
    for t in sorted(agent_trace.keys()): 
        # print out the time stamp
        trace_t = agent_trace[t]

        out_file.write("Time Step \n")
        out_file.write(str(t)+'\n')

        # print the agent state
        out_file.write("AGENT IS LOCATED AT: \n")
        out_file.write(str(trace_t['state'])+'\n')

        # print the agent color
        out_file.write("AGENT COLOR: \n")
        out_file.write(str(trace_t['color'])+'\n')

        # print the other agents in its bubble
        out_file.write("OTHER AGENTS IN BUBBLE ARE LOCATED AT: \n")
        [out_file.write(str(ag)+'\n') for ag in trace_t['agents_in_bubble']]

        # print the agents goal
        out_file.write("AGENT GOAL IS:\n")
        out_file.write(str((trace_t['goals']))+'\n')
        


        # print out oracle dict
        if t != list(sorted(agent_trace.keys()))[-1]:
            try: 
                trace_nxt = agent_trace[t+1]
            except:
                break

            # print out the oracle scores
            for ctrl, oracle_scores in trace_nxt['spec_struct_info'].items():
                out_file.write("control action\n")
                out_file.write(str(ctrl)+'\n')
            
                for key, value in oracle_scores.items():
                    out_file.write(key + ' ' + str(value)+'\n')

            out_file.write('\n')
            out_file.write("action selection strategy flags \n")
            out_file.write(str(trace_nxt['action_selection_flags'])+'\n')

            # straight action eval
            out_file.write("straight action evaluation \n")
            for ctrl, oracle_scores in trace_nxt['straight_action_eval'].items():
                out_file.write("control action\n")
                out_file.write(str(ctrl)+'\n')
                for key, value in oracle_scores.items():
                    out_file.write(key + ' ' + str(value)+'\n')

            out_file.write('\n')
        
            # print out the conflict requests it sent out
            out_file.write('sent requests to:\n')
            for agent in trace_nxt['sent_request']:
                out_file.write(str(agent)+'\n')
            
            out_file.write('received requests from\n')
            for agent in trace_nxt['received_requests']:
                out_file.write(str(agent)+'\n')

            out_file.write('agent token count after action is taken\n')
            out_file.write(str(trace_nxt['token_count'])+'\n')

            # print out max braking flag
            out_file.write('max braking flag sent\n')
            out_file.write(str(trace_nxt['max_braking_not_enough'])+'\n')

            # print out the agent intention
            out_file.write('agent intended action')
            out_file.write(str(trace_nxt['intention'])+'\n')

            # print control action taken
            out_file.write('control action taken\n')
            out_file.write(str(trace_nxt['action'])+'\n')

        out_file.write('\n')
        out_file.write('\n')
        pass
    out_file.close()

# test out the debug file 
if __name__ == '__main__':
    #output_dir = os.getcwd()+'/imgs/'
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    traces_file = os.getcwd()+'/saved_traces/game.p'

    outfile = os.getcwd()+'/saved_traces/debug.txt'
    print_one_agent_trace(traces_file, outfile, 12, 35, 'east', 46)

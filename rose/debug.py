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

# print out all agents and the conflict at a given time
def print_all_agents_and_conflict_requests_at_time_t(filename, outfile, time_step=None):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)

    out_file = open(outfile,"w")
    agents = traces[time_step]['agents']

    for agent in agents:
        x, y, theta, v, color, bubble, ag_id = agent
        tup = (x, y, theta, v, color, ag_id)
        print("======================AGENT IS LOCATED ATTTT==================")
        print(str(tup))
    
        try: 
            #print(str(tup))
            agent_info = traces[int(ag_id)][time_step]
            agents = agent_info['agents_in_bubble']
            print("agents in bubble")
            for agent in agents: 
                print(agent)
            #for key, value in agent_info.items():
            #    print(key)
            #print("agents in bubble")
            #agents = agent_info['agents_in_bubble']
            #for agent in agents: 
            #    print(agent)

            #out_file.write("======================AGENT IS LOCATED ATTTT================== \n")
            #out_file.write(str(tup)+'\n')
        except:
            break
        
        out_file.write("agent intention is:\n")
        out_file.write(str(agent_info['intention'])+'\n')

        # print the other agents in its bubble
        out_file.write("other agents in bubble located at: \n")
        [out_file.write(str(ag)+'\n') for ag in agent_info['agents_in_bubble']]

        # print the other agents in its bubble
        out_file.write("other agents in bubble before request located at: \n")
        [out_file.write(str(ag)+'\n') for ag in agent_info['agents_in_bubble_before']]

        out_file.write("sent requests to \n")
        for agent in agent_info['sent']:
            out_file.write(str(agent)+'\n')
            #out_file.write(str(agent)+'\n')
        
        out_file.write('received requests from\n')
        #print("received requests from")
        for agent in agent_info['received']:
            #print(agent)
            out_file.write(str(agent)+'\n')
        
        out_file.write('checked for agent conflict with:\n')
        #print("checked agent conflict with")
        for agent in agent_info['checked_for_conflict']:
            #print(agent)
            out_file.write(str(agent)+'\n')

def print_one_agent_trace(filename, outfile, x, y, heading, t):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    out_file = open(outfile,"w")

    # select an agent at random
    agent_id = get_agent_id(filename, x, y, heading, t)
    agent_trace = traces[agent_id].copy()

    # get agent params then remove from trace
    agent_param = agent_trace['agent_param']
    del agent_trace['agent_param']
    t_end = traces['t_end']

    #print(agent_trace.keys())
    # inspect the agent trace over time (how the agent is making it's decisions)
    for t in sorted(agent_trace.keys()):
        # print out the time stamp
        trace_t = agent_trace[t]

        out_file.write("Time Step \n")
        out_file.write(str(t)+'\n')

        # print the agent id
        out_file.write("AGENT IS LOCATED AT: \n")
        out_file.write(str(trace_t['agent_id'])+'\n')

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
            out_file.write('left turn gap info:\n')
            for tup in trace_nxt['left_turn_gap_arr']:
                out_file.write(str(tup)+'\n')

            out_file.write('\n')

            # print out which agents it checked conflict with
            out_file.write('checked for agent conflict with:\n')
            for agent in trace_nxt['checked_for_conflict']:
                out_file.write(str(agent)+'\n')

            # print out the conflict requests it sent out
            out_file.write('sent requests to:\n')
            for agent in trace_nxt['sent']:
                out_file.write(str(agent)+'\n')

            out_file.write('received requests from\n')
            for agent in trace_nxt['received']:
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
    #print_one_agent_trace(traces_file, outfile, 17,2, 'east', 2)

    outfile_cc = os.getcwd()+'/saved_traces/debug_cc.txt'
    print_all_agents_and_conflict_requests_at_time_t(traces_file, outfile_cc, 5)

import os
import _pickle as pickle
import random
import copy as cp


def write_info(out_file, t, trace):
    out_file.write("======================TIME STEP AT ==================\n")
    out_file.write(str(t)+'\n')
    
    # print the agent id
    out_file.write("AGENT IS LOCATED AT: \n")
    out_file.write(str(trace['agent_id'])+'\n')

    # print the agent state
    out_file.write("AGENT IS LOCATED AT: \n")
    out_file.write(str(trace['state'])+'\n')

    # print the agent color
    out_file.write("AGENT COLOR: \n")
    out_file.write(str(trace['color'])+'\n')

    # agent token count before resolution
    out_file.write("AGENT TOKEN COUNT BEFORE: \n")
    out_file.write(str(trace['token_count_before'])+'\n')

    # agent token count after resolutoin
    out_file.write("AGENT TOKEN COUNT AFTER: \n")
    out_file.write(str(trace['token_count'])+'\n')

    # print the other agents in its bubble
    out_file.write("OTHER AGENTS IN BUBBLE ARE LOCATED AT: \n")
    [out_file.write(str(ag)+'\n') for ag in trace['agents_in_bubble']]

    # print the agents goal
    out_file.write("AGENT GOAL IS:\n")
    out_file.write(str((trace['goals']))+'\n')

    # print the agent's subgoals
    out_file.write("AGENT SUBGOALS IS:\n")
    out_file.write(str((trace['subgoals']))+'\n')


    # print out the oracle scores
    for ctrl, oracle_scores in trace['spec_struct_info'].items():
        out_file.write("control action\n")
        out_file.write(str(ctrl)+'\n')

        for key, value in oracle_scores.items():
            #print('oracle')
            out_file.write(key + ' ' + str(value)+'\n')

    out_file.write('\n')
    out_file.write("action selection strategy flags \n")
    out_file.write(str(trace['action_selection_flags'])+'\n')

    # straight action eval
    out_file.write("straight action evaluation \n")
    for ctrl, oracle_scores in trace['straight_action_eval'].items():
        out_file.write("control action\n")
        out_file.write(str(ctrl)+'\n')
        for key, value in oracle_scores.items():
            out_file.write(key + ' ' + str(value)+'\n')
    
    # lead agent found for backup plan evaluation
    out_file.write("lead agent is:\n")
    out_file.write(str(trace['lead_agent'])+'\n')

    out_file.write('\n')
    out_file.write('left turn gap info:\n')
    for tup in trace['left_turn_gap_arr']:
        out_file.write(str(tup)+'\n')
    out_file.write('\n')

    out_file.write('\n')
    out_file.write('clearance straight info :\n')
    out_file.write(str(trace['clearance_straight_info'])+'\n')
    out_file.write('\n')

    # print out which agents it checked conflict with
    out_file.write('checked for agent conflicts with:\n')
    for agent in trace['checked_for_conflict']:
        out_file.write(str(agent)+'\n')

    # print out the conflict requests it sent out
    out_file.write('sent requests to:\n')
    for agent in trace['sent']:
        out_file.write(str(agent)+'\n')

    out_file.write('received requests from\n')
    for agent in trace['received']:
        out_file.write(str(agent)+'\n')

    out_file.write('agent token count after action is taken\n')
    out_file.write(str(trace['token_count'])+'\n')

    # print out max braking flag
    out_file.write('max braking flag sent\n')
    out_file.write(str(trace['max_braking_not_enough'])+'\n')

    # print out the agent intention
    out_file.write('agent intended action')
    out_file.write(str(trace['intention'])+'\n')

    # print control action taken
    out_file.write('control action taken\n')
    out_file.write(str(trace['action'])+'\n')

    out_file.write('\n')
    out_file.write('\n')


def print_one_agent_trace(filename, x, y, heading, time, outfile):
    def get_agent_id(traces, x, y, heading, t):
        agent_dict = traces[t]
        #print(agent_dict)
        for agent_id, agent_trace in agent_dict.items():
            agent_state = agent_trace['state']
            if agent_state is not None: 
                x_ag, y_ag, heading_ag, v_ag = agent_state
                if heading_ag == heading:
                    color_ag = agent_trace['color']
                    print(x_ag, y_ag, heading_ag, color_ag)
                if (x_ag, y_ag, heading_ag) == (x, y, heading):
                    return agent_id

        print("Agent is not found!")
        return None

    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)
    
    out_file = open(outfile,"w")
    
    # look for the agent id of the specified agent 
    agent_id = get_agent_id(traces, x, y, heading, time)

    t_end = int(max(traces.keys()))
    #print(t_end)
    for t in range(1,t_end):
        agents_dict = traces[t]
        # search through all agents at time t
        for ag_id, agent_trace in agents_dict.items():
            # if agent is at time t, print out information
            if ag_id == agent_id: 
                # print information
                write_info(out_file, t, agent_trace)

    out_file.close()


# print out all agents and the conflict at a given time
def print_all_agents_at_time_t(filename, outfile, time_step=None):
    with open(filename, 'rb') as pckl_file:
        traces = pickle.load(pckl_file)

    out_file = open(outfile,"w")
    agent_dict = traces[time_step]
    # loop through all agents at time t
    for agent_id, agent_trace in agent_dict.items():
        out_file.write("======================AGENT IS LOCATED ATTTT==================\n")
        out_file.write(str(agent_trace['state'])+'\n')
        out_file.write(str(agent_trace['color'])+'\n')

        out_file.write("AGENT GOAL IS:\n")
        out_file.write(str((agent_trace['goals']))+'\n')

        out_file.write("AGENT SUBGOALS IS:\n")
        out_file.write(str((agent_trace['subgoals']))+'\n')

        out_file.write("agent intention is:\n")
        out_file.write(str(agent_trace['intention'])+'\n')

        # agent token count before resolution
        out_file.write("AGENT TOKEN COUNT BEFORE: \n")
        out_file.write(str(agent_trace['token_count_before'])+'\n')

        # agent token count after resolutoin
        out_file.write("AGENT TOKEN COUNT AFTER: \n")
        out_file.write(str(agent_trace['token_count'])+'\n')

        # print the other agents in its bubble
        out_file.write("other agents in bubble before request located at: \n")
        [out_file.write(str(ag)+'\n') for ag in agent_trace['agents_in_bubble_before']]

        out_file.write("sent requests to \n")
        for agent in agent_trace['sent']:
            out_file.write(str(agent)+'\n')
        
        out_file.write('received requests from\n')
        for agent in agent_trace['received']:
            out_file.write(str(agent)+'\n')
        
        out_file.write("ACTION SELECTION STRATEGY FLAGS: \n")
        out_file.write(str(agent_trace['action_selection_flags'])+'\n')

        out_file.write("ACTION TAKEN: \n")
        out_file.write(str(agent_trace['action'])+'\n')
        
        out_file.write('checked for agent conflict with:\n')
        if agent_trace['checked_for_conflict'] is not None: 
            for agent in agent_trace['checked_for_conflict']:
                out_file.write(str(agent)+'\n')

        out_file.write('agent max braking flag :\n')
        out_file.write(str(agent_trace['max_braking_not_enough'])+'\n')

        out_file.write('conflict cluster winner:\n')
        out_file.write(str(agent_trace['conflict_winner'])+'\n')

        out_file.write('agent ')


# test out the debug file
if __name__ == '__main__':
    traces_file = os.getcwd()+'/saved_traces/game_debug.p'
    outfile = os.getcwd()+'/saved_traces/debug.txt'
    #print_one_agent_trace(traces_file, 24, 65, 'west', 67, outfile)
    print_one_agent_trace(traces_file, 31, 46, 'west', 32, outfile)

    #outfile_cc = os.getcwd()+'/saved_traces/debug_cc.txt'
    #print_all_agents_at_time_t(traces_file, outfile_cc, 108)

#%% IMPORT FUNCTIONS
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import networkx as nx
import datetime
import os
#import user-defined functions
from functions import define_grid_graph,define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, calculate_network_resistance, update_edge_weigths, node_map

#%% MAKE OUTPUT DIRECTORY

folder = r'./output/' 
if not os.path.exists(folder):
    os.makedirs(folder)
    
subfolder = r'./output/network_model/' 
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
#%% USER SETUP

########## Define grid graph structure ##########
rows = 21 # number of nodes' rows of the graph
cols = 21 # number of nodes' columns of the graph
random_diag = 1 # 1 = with random diagonals, 0 = without random diagonals
seed = 2 # define the random seed for the random diagonals (if random_diag=0 it is useless)

kp0 = 2.555173332603108574e-06 # model kp_0
kd0 = 6.488388862524891465e+01 # model kd_0
eta_p = 3.492155165334443012e+01 # model eta_p
eta_d = 5.590601016803570467e+00 # model eta_d
g_min = 1.014708121672117710e-03 # model g_min
g_max = 2.723493729125820492e-03 # model g_max
g0 = g_min # model g_0
#################################################

########## Define source and ground pads lists ##########
src = [node_map(10, 1, rows, cols), 85] # define a list of source nodes in the range [0, rows*cols-1]
gnd = [409] # define a list of ground nodes in the range [0, rows*cols-1]
#########################################################

########## Define input voltage and time lists ##########
V_list = [[] for s in range(len(src))] # V_list[i] is the input on src[i]

V_list[0] = [0.1]*10 + ['f']*20 + [0.1]*10 # [V]
V_list[1] = [6]*10 + [6]*20 + [0.1]*10 # [V]

t_list = list(np.linspace(0, 20e-3, 40)) # [s]
#########################################################

########### Define 2-tuples of nodes for conductance reading #####
Y_node_read = [(23,56), (57, 79), (92, 58)] # list of 2-tuples nodes in the range [0, rows*cols-1]
V_node_read = [5, 200, 45] # list of nodes in the range [0, rows*cols-1]
##################################################################

anim_plot = 1 # 1 to plot animation, 0 otherwise
anim_save = 0 # 1 to save animation, 0 otherwise (if 1, 'anim_plot' must be 1 too)
save_data = 1 # 1 to save data, 0 otherwise

#########################################################
#%% ERROR CHECK

if random_diag not in [0,1]:
    raise Exception('Error: "random_diag" not valid, must be 1 or 0.')

if len(src) == 0:
    raise Exception('Error: "src" length must not be zero.')
if len(gnd) == 0:
    raise Exception('Error: "gnd" length must not be zero.')
    
for s in src:
    if s<0 or s>(rows*cols-1):
        raise Exception('Error: "src['+str(src.index(s))+']" out of range.')
        
for g in gnd:
    if g<0 or g>(rows*cols-1):
        raise Exception('Error: "gnd['+str(gnd.index(g))+']" out of range.')
        
if len(V_list) != len(src):
    raise Exception('Error: "V_list" length must coincide with "src" length.')
    
for s in range(len(src)):
    if len(V_list[s]) != len(t_list):
        raise Exception('Error: "V_list['+str(s)+']" length must coincide with "t_list" length.')

for nr in range(len(Y_node_read)):
    if len(Y_node_read[nr]) != 2:
       raise Exception('Error: "Y_node_read['+str(nr)+']" must be a 2-tuple.')
    for i in range(2):
        if Y_node_read[nr][i]<0 or Y_node_read[nr][i]>(rows*cols-1):
            raise Exception('Error: "Y_node_read['+str(nr)+']['+str(i)+']" out of range.') 

for nr in range(len(V_node_read)):
    if V_node_read[nr]<0 or V_node_read[nr]>(rows*cols-1):
        raise Exception('Error: "V_node_read['+str(nr)+']['+str(i)+']" out of range.') 

if anim_plot not in [0,1]:
    raise Exception('Error: "anim_plot" not valid, must be 1 or 0.')
if anim_save not in [0,1]:
    raise Exception('Error: "anim_save" not valid, must be 1 or 0.')
if save_data not in [0,1]:
    raise Exception('Error: "save_data" not valid, must be 1 or 0.')
#################################################################################

#%% GRAPH INTIALIZATION

if random_diag == 0:
    G = define_grid_graph(rows, cols) # grid graph without random diagonals
elif random_diag == 1:
    G = define_grid_graph_2(rows, cols, seed) # grid graph with random diagonals

initialize_graph_attributes(G, g0) # initialize each edge with 'g0' value

#########################################################################

#%% PRINT SIMULATION SETUP

all_input = [float(V_list[s][t]) for s in range(len(src)) for t in range(len(t_list)) if V_list[s][t] != 'f']

print('Grid: '+str(rows)+'x'+str(cols))
print('Random diagonals: '+ (1-random_diag)*'NO' + random_diag*'YES'+random_diag*' (seed='+random_diag*str(seed)+random_diag*')')
print('Timesteps: '+str(len(t_list)))
print('Sources number: '+str(len(src)))
print('Grounds number: '+str(len(gnd)))
print('Minimum input voltage: ' +str(min(all_input))+'V')
print('Maximum input voltage: ' +str(max(all_input))+'V')
print('Time start: '+str(t_list[0])+'s')
print('Time stop: '+str(t_list[-1])+'s')

#%% GRAPH EVOLUTION

H_list = [[] for t in range(len(t_list))]

H_list[0] = mod_voltage_node_analysis(G, [V_list[s][0] for s in range(len(src))], src, gnd)

print('\n')
sys.stdout.write("\rNetwork Stimulation: "+str(1)+'/'+str(len(t_list)))

for t in range(1, len(t_list)):
    
    delta_t = t_list[t] - t_list[t-1]     
    G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)
    H_list[t] = mod_voltage_node_analysis(G, [V_list[s][t] for s in range(len(src))], src, gnd)

    sys.stdout.write("\rNetwork Stimulation: "+str(t+1)+'/'+str(len(t_list)))        
##########################################################################

#%% CONDUCTANCE and VOLTAGE: READ & PLOT
    
Y_out = [[] for nr in range(len(Y_node_read))]
V_out = [[] for nr in range(len(V_node_read))]

print('\n')
for t in range(len(t_list)):
    for nr in range(len(Y_node_read)):
        Y_out[nr] += [1/calculate_network_resistance(H_list[t], [Y_node_read[nr][0]], [Y_node_read[nr][1]])]
    for nr in range(len(V_node_read)):
        V_out[nr] += [H_list[t].nodes[V_node_read[nr]]['V']]
    sys.stdout.write("\rOutput Reading: "+str(t+1)+'/'+str(len(t_list)))  

plt.figure()
plt.title('Voltage Input', fontsize = 25)
for v in range(len(V_list)):
    float_index = [i for i in range(len(V_list[v])) if V_list[v][i] == 'f']
    value_index = [i for i in range(len(V_list[v])) if V_list[v][i] != 'f']
    p = plt.plot([t_list[i] for i in value_index], [float(V_list[v][i]) for i in value_index], label='Node '+str(src[v]), linewidth=2)
    color = p[0].get_color()
    plt.plot([t_list[i] for i in float_index], [0]*len(float_index), 'x', color=color, linewidth = 2)
plt.xlabel('Time [s]', fontsize = 20)
plt.ylabel('Voltage [V]', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()

plt.figure()
plt.title('Conductance Reading', fontsize = 25)
for nr in range(len(Y_node_read)):
    plt.plot(t_list, Y_out[nr], label='Nodes '+str(Y_node_read[nr]), linewidth=2)
plt.xlabel('Time [s]', fontsize = 20)
plt.ylabel('Conductance [S]', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()

plt.figure()
plt.title('Voltage Reading', fontsize = 25)
for nr in range(len(V_node_read)):
    plt.plot(t_list, V_out[nr], label='Node '+str(V_node_read[nr]), linewidth=2)
plt.xlabel('Time [s]', fontsize = 20)
plt.ylabel('Voltage [V]', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()

#%% ANIMATION PLOT

if anim_plot == 1:
    frames_num = len(t_list)
    frames_interval = 100
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(i):
    
        plt.cla()
        pos=nx.get_node_attributes(H_list[i],'pos')
    
        nx.draw_networkx(H_list[i], pos,
                          #NODES
                          node_size=60,
                          node_color=[H_list[i].nodes[n]['V'] for n in H_list[i].nodes()],
                          cmap=plt.cm.Blues,
                          vmin=0,
                          vmax=max(all_input),
                          #EDGES
                          width=4,
                          edge_color=[H_list[i][u][v]['Y'] for u,v in H_list[i].edges()],
                          edge_cmap=plt.cm.Reds,
                          edge_vmin=g_min,
                          edge_vmax=g_max,
                          with_labels=False,   #Set TRUE to see node numbers
                          font_size=6,)
    
        nx.draw_networkx_nodes(H_list[i], pos, nodelist=src+gnd, node_size=100, node_color='k')
        plt.title('t = '+str(round(t_list[i], 4))+' s', fontsize=30)
    
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=frames_num, interval=frames_interval, blit=False, repeat=True)
    
    if anim_save == 1:
        print('\n')
        print('Animation Saving...')
        current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
        current_date_and_time_string = str(current_date_and_time)
        file_name =  current_date_and_time_string
        anim.save('./output/network_model/'+file_name+'_animation.gif', writer='imagemagick')
#%% OUTPUT FILE SAVING
if save_data == 1:
        
    original_stdout = sys.stdout
       
    current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    current_date_and_time_string = str(current_date_and_time)
    extension = ".txt"
    file_name =  current_date_and_time_string + extension
    file = open('./output/network_model/'+file_name, 'w')
    
    sys.stdout = file 
    print('SIMULATION SETUP\n')
    print('Grid: '+str(rows)+'x'+str(cols))
    print('Random diagonals: '+ (1-random_diag)*'NO' + random_diag*'YES'+random_diag*' (seed='+random_diag*str(seed)+random_diag*')')
    print('Timesteps: '+str(len(t_list)))
    print('Sources number: '+str(len(src))+' '+str(src))
    print('Grounds number: '+str(len(gnd))+' '+str(gnd))
    print('Minimum input voltage: ' +str(min(all_input))+'V')
    print('Maximum input voltage: ' +str(max(all_input))+'V')
    print('Time start: '+str(t_list[0])+'s')
    print('Time stop: '+str(t_list[-1])+'s') 
    print('\n')
    print('INPUT/OUTPUT DATA\n')
    header = 'Time'
    for i in range(len(src)):
        header += ' V_in_node'+str(src[i])+'\t'
    for i in range(len(V_node_read)):
        header += ' V_out_node'+str(V_node_read[i])+'\t'
    for i in range(len(Y_node_read)):
        header += ' Y_out_nodes'+str(Y_node_read[i])+'\t'   
    print(header[:-1]+'\n')
    for l in range(len(t_list)):
        line = [t_list[l]]
        for i in range(len(src)):
            if V_list[i][l]=='f':
                line += [str(V_list[i][l])]
            elif V_list[i][l]!='f':
                line += [V_list[i][l]]
        for i in range(len(V_node_read)):
            line += [V_out[i][l]]
        for i in range(len(Y_node_read)):
            line += [Y_out[i][l]]
        for element in line:
            print(element, end='\t')
        print('')
    
    file.close()
    sys.stdout = original_stdout 
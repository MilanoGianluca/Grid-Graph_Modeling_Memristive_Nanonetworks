#%% IMPORT FUNCTIONS
import matplotlib.pyplot as plt
import matplotlib.animation
import sys, os
import datetime
import networkx as nx
import numpy as np
from functions import define_grid_graph,define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, calculate_network_resistance, calculate_Vsource, calculate_Isource, update_edge_weigths 
from file_sel import file_sel

#%% MAKE OUTPUT DIRECTORY

folder = r'./output/' 
if not os.path.exists(folder):
    os.makedirs(folder)
    
subfolder = r'./output/fit_testing/' 
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
    
#%% USER SETUP

fdir = './raw_data_fitting/' # directory of file containing raw data to fit
fname = 'example_file_1.txt' # name of the file containing raw data to fit

SR = 0  # keep the same sample rate (or ad hoc modified in "file_sel") with SR = 1 (where provided)
step = 15 # if implemented in "file_sel", points are sampled one each step

########## Define grid graph structure ##########
rows = 21 # number of nodes' rows of the graph
cols = 21 # number of nodes' columns of the graph
random_diag = 1 # 1 = with random diagonals, 0 = without random diagonals
seed = 2 # define the random seed for the random diagonals (if random_diag=0 it is useless)

src = 31 # define source node position (may be indexed with the function 'node_map(x, y, rows, cols)')
gnd = 409 # define ground node position (may be indexed with the function 'node_map(x, y, rows, cols)')

plot_gif = 0 # 1 to plot animation, 0 otherwise
save_gif = 0 # 1 to save animation, 0 otherwise (if 1, 'plot_gif' must be 1 too)
save_data = 1 # 1 to save data, 0 otherwise
#%% ERROR CHECK

if random_diag not in [0,1]:
    raise Exception('Error: "random_diag" not valid, must be 1 or 0.')

if src<0 or src>(rows*cols-1):
    raise Exception('Error: "src out of range.')
        
if gnd<0 or gnd>(rows*cols-1):
    raise Exception('Error: "gnd out of range.')
    
#%% IMPORT & PLOT RAW DATA TO FIT 

time, V, I_exp, title, param = file_sel(fdir, fname, SR, step)
G_exp = I_exp/V

plt.figure(figsize=(10, 10))
plt.suptitle('Experimental data - '+title, fontsize = 20)
plt.subplot(211)
ax1 = plt.gca()
plt.grid()
ax1.set_xlabel('time [s]', fontsize = 15)
ax1.set_ylabel('Voltage [V]', color='blue', fontsize = 15)
plt.plot(time, V, 'b', linewidth = 1.5)
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Current [A]', color='red', fontsize = 15)
plt.plot(time, I_exp, 'r--', linewidth = 1.5)
ax2.tick_params(axis='y', labelcolor='red')

plt.subplot(212)
plt.grid()
plt.plot(time, G_exp, 'b', linewidth = 1.5)
plt.xlabel('time [s]', fontsize = 15)
plt.ylabel('Conductance [S]', fontsize = 15)

#%% PRINT SIMULATION SETUP

print('Grid: '+str(rows)+'x'+str(cols))
print('Random diagonals: '+ (1-random_diag)*'NO' + random_diag*'YES'+random_diag*' (seed='+random_diag*str(seed)+random_diag*')')
print('Timesteps: '+str(len(time)))
print('Source index: '+str(src))
print('Ground index: '+str(gnd))
print('Minimum experimental voltage: ' +str(np.min(V))+'V')
print('Maximum experimental voltage: ' +str(np.max(V))+'V')
print('Time start: '+str(time[0])+'s')
print('Time stop: '+str(time[-1])+'s')

#%% NETWORK STIMULATION

# Graph initialization
if random_diag == 0:
    G = define_grid_graph(rows, cols) # grid graph without random diagonals
elif random_diag == 1:
    G = define_grid_graph_2(rows, cols, seed) # grid graph with random diagonals

kp0 = param[0]
kd0 = param[2]
eta_p = param[1]
eta_d = param[3]
g0 = param[4]
g_min = param[5]
g_max = param[6]


G = initialize_graph_attributes(G, g0)

timesteps = len(time)
Vin_list = list(V)
t_list = list(time)

# Initialization of list over time
H_list = [[] for t in range(0, timesteps)]
I_list = [[] for t in range(0, timesteps)]
V_list = [[] for t in range(0, timesteps)]
Ynetwork_list = [[] for t in range(0, timesteps)]
Rnetwork_list = [[] for t in range(0, timesteps)]

# Pristine state
H_list[0] = mod_voltage_node_analysis(G, [V[0]], [src], [gnd])
V_list[0] = calculate_Vsource(H_list[0], src)
I_list[0] = calculate_Isource(H_list[0], src)
Rnetwork_list[0] = calculate_network_resistance(H_list[0], [src], [gnd])
Ynetwork_list[0] = 1/Rnetwork_list[0]

print('\n')
sys.stdout.write("\rNetwork Stimulation: "+str(1)+'/'+str(len(t_list)))

# Growth over time
for i in range(1, int(timesteps)):

    delta_t = t_list[i] - t_list[i-1]

    update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
    
    H_list[i] = mod_voltage_node_analysis(G, [V[i]], [src], [gnd])
    V_list[i] = calculate_Vsource(H_list[i], src)
    I_list[i] = calculate_Isource(H_list[i], src)
    Rnetwork_list[i] = calculate_network_resistance(H_list[i], [src], [gnd])
    Ynetwork_list[i] = 1/Rnetwork_list[i]
    
    sys.stdout.write("\rNetwork Stimulation: "+str(i+1)+'/'+str(len(t_list)))   
#%% PLOTS

# Conductance
plt.figure(figsize=(10,7))
plt.title(title, fontsize=30)
plt.plot(t_list, G_exp, 'b', label='exp', linewidth = 2.2)
plt.plot(t_list, Ynetwork_list, '--r', label='model', linewidth = 2.2)
plt.legend(fontsize=30)
plt.grid()
plt.xlabel('time [s]', fontsize=30)
plt.ylabel('Conductance [S]', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

#%% ANIMATION
frame_step = 1 # plot one frame each frame_step
frames_num = timesteps//frame_step
frames_interval = 150 

if plot_gif == 1:
    fig, ax = plt.subplots(figsize=(10, 10))
    def update(j):
    
        plt.cla()
        i = frame_step*j
        
        pos=nx.get_node_attributes(H_list[i],'pos')
    
        nx.draw_networkx(H_list[i], pos,
                         #NODES
                         node_size=60,
                         node_color=[H_list[i].nodes[n]['V'] for n in H_list[i].nodes()],
                         cmap=plt.cm.Blues,
                         vmin=np.min(V),
                         vmax=np.max(V),
                         #EDGES
                         width=4,
                         edge_color=[H_list[i][u][v]['Y'] for u,v in H_list[i].edges()],
                         edge_cmap=plt.cm.Reds,
                         edge_vmin=g_min,
                         edge_vmax=g_max,
                         with_labels=False,   #Set TRUE to see node numbers
                         font_size=6,)
        nx.draw_networkx_nodes(H_list[i], pos, nodelist=[src, gnd], node_size=100, node_color='k')
        plt.title('t = '+str(round(t_list[i],2))+' s', fontsize=25)
    
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=frames_num, interval=frames_interval, blit=False, repeat=True)

    if save_gif==1:
        print('\n')
        print('Animation Saving...')
        current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
        current_date_and_time_string = str(current_date_and_time)
        file_name = './output/fit_testing/'+current_date_and_time_string+'_'+title+'.txt'
        anim.save(file_name+'_animation.gif', writer='imagemagick')
#%% Export Data

if save_data == 1:
    original_stdout = sys.stdout
    current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    current_date_and_time_string = str(current_date_and_time)
    file_name = current_date_and_time_string+'_fit_test_'+title+'.txt'
    file = open('./output/fit_testing/'+file_name, 'w')
    sys.stdout = file    
    print('Fit test from file: '+title)
    print('\n')
    print('FIT TEST SETUP')
    print('Grid: '+str(rows)+'x'+str(cols))
    print('Random diagonals: '+ (1-random_diag)*'NO' + random_diag*'YES'+random_diag*' (seed='+random_diag*str(seed)+random_diag*')')
    print('Timesteps: '+str(len(time))+' (step: '+str(step)+', SR: '+(1-SR)*'NO' + SR*'YES'+')')
    print('Source index: '+str(src))
    print('Ground index: '+str(gnd))
    print('Minimum experimental voltage: ' +str(min(V))+'V')
    print('Maximum experimental voltage: ' +str(max(V))+'V')
    print('Time start: '+str(time[0])+'s')
    print('Time stop: '+str(time[-1])+'s')
    print('\n')
    current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    current_date_and_time_string = str(current_date_and_time)
    # file_name = './output/fit_testing/'+current_date_and_time_string+'_'+title+'.txt'
    my_data=np.vstack((t_list, V_list, list(I_exp), list(G_exp), I_list, Ynetwork_list))
    my_data=my_data.T
    print('time, V, I_exp, G_exp, I_model, G_model')
    np.savetxt(file, my_data, delimiter='\t')#, header='time, V, I_exp, G_exp, I_model, G_model',comments='')
    file.close()
    sys.stdout = original_stdout 
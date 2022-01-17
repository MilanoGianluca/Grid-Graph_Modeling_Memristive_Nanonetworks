#%% IMPORT FUNCTIONS
import matplotlib.pyplot as plt
from numpy import zeros, ones
from scipy.optimize import curve_fit
import warnings
import os
import sys
import datetime
from functions import define_grid_graph,define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, calculate_network_resistance, update_edge_weigths 
from file_sel import file_sel

#%% MAKE OUTPUT DIRECTORY

folder = r'./output/' 
if not os.path.exists(folder):
    os.makedirs(folder)
    
subfolder = r'./output/fitting_conductance/' 
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
    
#%% USER SETUP

fdir = './raw_data_fitting/' # directory of file containing raw data to fit
fname = 'example_file_1.txt' # name of the file containing raw data to fit

# Choose the variable to fit: 1=yes, 0=no (N.B. the fixed value, eventually, is its starting point)
sel_kp0 = 1
sel_kd0 = 1
sel_eta_p = 1
sel_eta_d = 1
sel_g_min = 1
sel_g_max = 1

max_val = 1e5  # fitting parameters are found in the range [0, max_val]
maxfev = 400 # maximum function evalutation during fitting
ftol = 5e-6 # tolerance for fitting to stop

SR = 0  # keep the same sample rate (or ad hoc modified in "file_sel") with SR = 1 (where provided)
step = 15 # if implemented in "file_sel", points are sampled one each step

########## Define grid graph structure ##########
rows = 21 # number of nodes' rows of the graph
cols = 21 # number of nodes' columns of the graph
random_diag = 1 # 1 = with random diagonals, 0 = without random diagonals
seed = 2 # define the random seed for the random diagonals (if random_diag=0 it is useless)

src = 31 # define source node position (may be indexed with the function 'node_map(x, y, rows, cols)')
gnd = 409 # define ground node position (may be indexed with the function 'node_map(x, y, rows, cols)')

var = [sel_kp0, sel_kd0, sel_eta_p, sel_eta_d, sel_g_min, sel_g_max]

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

plt.figure()
plt.suptitle('Experimental data - '+title)
plt.subplot(211)
ax1 = plt.gca()
plt.grid()
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Voltage [V]', color='blue')
plt.plot(time, V, 'b')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Current [A]', color='red')
plt.plot(time, I_exp, 'r--')
ax2.tick_params(axis='y', labelcolor='red')

plt.subplot(212)
plt.grid()
plt.plot(time, G_exp, 'b')
plt.xlabel('time [s]')
plt.ylabel('Conductance [S]')

#%% PRINT SIMULATION SETUP

print('Grid: '+str(rows)+'x'+str(cols))
print('Random diagonals: '+ (1-random_diag)*'NO' + random_diag*'YES'+random_diag*' (seed='+random_diag*str(seed)+random_diag*')')
print('Timesteps: '+str(len(time))+' (step: '+str(step)+')')
print('Source index: '+str(src))
print('Ground index: '+str(gnd))
print('Minimum experimental voltage: ' +str(min(V))+'V')
print('Maximum experimental voltage: ' +str(max(V))+'V')
print('Time start: '+str(time[0])+'s')
print('Time stop: '+str(time[-1])+'s')
print('\n')
print('...Fitting...')
print('\n')
#%% FITTING 

# Graph initialization
if random_diag == 0:
    Gdef = define_grid_graph(rows, cols) # grid graph without random diagonals
elif random_diag == 1:
    Gdef = define_grid_graph_2(rows, cols, seed) # grid graph with random diagonals

# Initialization of list over time
timesteps = len(time)
H_list = [[] for t in range(0, timesteps)]

def init_G(t, gin):

    # t = time[0]
    Gpad = initialize_graph_attributes(Gdef, gin)
    H = mod_voltage_node_analysis(Gpad, [V[0]], [src], [gnd])
    R = calculate_network_resistance(H, [src], [gnd])
    Y = 1/R

    return Y

warnings.filterwarnings("ignore")
gg_0, _ = curve_fit(init_G, time[0], G_exp[0], p0=1e-4, method='trf', bounds=(0, 1e-2), maxfev=maxfev)
warnings.filterwarnings("default")

def model_grid(time, kp0, kd0, eta_p, eta_d, g_min, g_max):

    t_list = time

    Ynetwork_list = zeros(len(time), )
    Rnetwork_list = zeros(len(time), )

    G = initialize_graph_attributes(Gdef, gg_0)
    H_list[0] = mod_voltage_node_analysis(G, [V[0]], [src], [gnd])
    Rnetwork_list[0] = calculate_network_resistance(H_list[0], [src], [gnd])
    Ynetwork_list[0] = 1 / Rnetwork_list[0]

    for i in range(1, int(timesteps)):

        delta_t = t_list[i] - t_list[i - 1]

        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)
        
        H_list[i] = mod_voltage_node_analysis(G, [V[i]], [src], [gnd])
        Rnetwork_list[i] = calculate_network_resistance(H_list[i], [src], [gnd])
        Ynetwork_list[i] = 1 / Rnetwork_list[i]

    return Ynetwork_list

start_kp0 = param[0]
start_kd0 = param[2]
start_eta_p = param[1]
start_eta_d = param[3]
start_g_min = param[5]
start_g_max = param[6]

init_val = [start_kp0, start_kd0, start_eta_p, start_eta_d, start_g_min, start_g_max]

err_i = 1e-10*ones(len(var))
err_f = 1e-10*ones(len(var))

for i in range(len(var)):
    if var[i] == 1:
        err_i[i] = init_val[i]
        err_f[i] = max_val
        
best_val, covar = curve_fit(model_grid, time, G_exp, p0=init_val, method='trf',
                            bounds=((start_kp0-err_i[0], start_kd0-err_i[1], start_eta_p-err_i[2], start_eta_d-err_i[3], start_g_min-err_i[4], start_g_max-err_i[5]),
                                    (start_kp0+err_f[0], start_kd0+err_f[1], start_eta_p+err_f[2], start_eta_d+err_f[3], start_g_min+err_f[4], start_g_max+err_f[5])),
                            maxfev=maxfev, ftol=ftol)

fit_kp0 = best_val[0]
fit_kd0 = best_val[1]
fit_eta_p = best_val[2]
fit_eta_d = best_val[3]
fit_g_min = best_val[4]
fit_g_max = best_val[5]
fit_g0 = gg_0[0]

#%% SAVE DATA
original_stdout = sys.stdout
current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
current_date_and_time_string = str(current_date_and_time)
file_name = current_date_and_time_string+'_fit_param_'+title+'.txt'
file = open('./output/fitting_conductance/'+file_name, 'w')
sys.stdout = file    
print('Fitting data from file: '+title)
print('\n')
print('FITTING SETUP')
print('Grid: '+str(rows)+'x'+str(cols))
print('Random diagonals: '+ (1-random_diag)*'NO' + random_diag*'YES'+random_diag*' (seed='+random_diag*str(seed)+random_diag*')')
print('Timesteps: '+str(len(time))+' (step: '+str(step)+', SR: '+(1-SR)*'NO' + SR*'YES'')')
print('Source index: '+str(src))
print('Ground index: '+str(gnd))
print('Minimum experimental voltage: ' +str(min(V))+'V')
print('Maximum experimental voltage: ' +str(max(V))+'V')
print('Time start: '+str(time[0])+'s')
print('Time stop: '+str(time[-1])+'s')
print('\n')
print('FITTING PARAMETERS')
print('kp0 = ', fit_kp0)
print('kd0 = ', fit_kd0)
print('eta_p = ', fit_eta_p)
print('eta_d = ', fit_eta_d)
print('g0 = ', fit_g0)
print('g_min = ', fit_g_min)
print('g_max = ', fit_g_max)
file.close()
sys.stdout = original_stdout 

#%% PLOT DATA
G_fit = model_grid(time, fit_kp0, fit_kd0, fit_eta_p, fit_eta_d, fit_g_min ,fit_g_max)

print(title + ', fitting values:')
print('kp0 = ', fit_kp0)
print('kd0 = ', fit_kd0)
print('eta_p = ', fit_eta_p)
print('eta_d = ', fit_eta_d)
print('g0 = ', fit_g0)
print('g_min = ', fit_g_min)
print('g_max = ', fit_g_max)
print('\n')

plt.figure()
plt.title(title, fontsize = 20)
plt.plot(time, G_exp*V, 'b', label='exp', linewidth = 1.5)
plt.plot(time, G_fit*V, '--r', label='model', linewidth = 1.5)
plt.legend(fontsize = 15)
plt.grid()
plt.xlabel('time [s]', fontsize = 15)
plt.ylabel('Current [A]', fontsize = 15)

plt.figure()
plt.title(title, fontsize = 20)
plt.plot(time, G_exp, 'b', label='exp', linewidth = 1.5)
plt.plot(time, G_fit, '--r', label='model', linewidth = 1.5)
plt.legend(fontsize = 15)
plt.grid()
plt.xlabel('time [s]', fontsize = 15)
plt.ylabel('Conductance [S]', fontsize = 15)



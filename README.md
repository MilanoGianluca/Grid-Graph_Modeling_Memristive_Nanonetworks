# Grid-Graph_Modeling_Memristive_Nanonetworks
This repository contains codes of the paper "Grid-graph modeling of emergent neuromorphic dynamics and heterosynaptic plasticity in memristive nanonetworks", Neuromorphic Computing and Engineering (2022), DOI: https://doi.org/10.1088/2634-4386/ac4d86.

The python scripts and functions in the folder allow to perform two main tasks: 
	
  1) Fit two-terminal experimental conductance curves associated to the time evolution 	of a memristive network. The model for each memristive edge of the network is a balanced rate equation [1] 
	
  2) Stimulate a customized memristive network with user-defined voltage streams 	associated to user-defined source and ground nodes. \

Output results are saved in output folder, which contains 3 sub-folders associated to the 3 main code scripts. Each saved file name begins with the current data and time to avoid overwriting of files. \
\
Files details are reported in the following. \
\
<ins>functions.py</ins> Python file including all the functions to define the grid graph network model and to perform the nodal analysis by means of the modified voltage node analysis algorithm. It also includes functions to perform reading of network features (resistance, voltage, current). \
\
<ins>file_sel.py</ins> Python file including the function to import data from file (imported from the folder raw_data_fitting) for fitting process. It allows to manage input file by proper sampling data before fitting them in the main script. \
\
<ins>fitting_conductance.py</ins> Python file which imports experimental data by means of the function file_sel.py and fits the 2-terminal data with the model defined in functions.py to extrapolate the seven model parameters. \
\
<ins>fit_testing.py</ins> Python file which allows to stimulate the network with the same data and parameters used in fitting_conductance.py (imported from file_sel.py). This script is useful to choose good starting points by-hand for fitting process and to verify and extrapolate extra quantities after fitting procedure. 
\
\
<ins>network_model.py</ins> Python script which allows to stimulate the network by defining an arbitrary source and ground nodes number and the associated voltage streams in time. Output quantities as voltage and conductance on chosen nodes are displayed and saved.

<h4> Reference </h4>
[1] Miranda, Enrique, Gianluca Milano, and Carlo Ricciardi. "Modeling of short-term synaptic plasticity effects in ZnO nanowire-based memristors using a potentiation-depression rate balance equation." IEEE Transactions on Nanotechnology 19 (2020): 609-612.

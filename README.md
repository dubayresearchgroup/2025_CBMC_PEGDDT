This repository contains the data and analysis scripts for the publication "Engineering Nanoparticle Surface Amphiphilicity: An Integrated Computational and Laser Desorption Ionization Study of Controlled Ligand Self-Assembly."

The directory "data" contains text files containing data obtained experimentally and computationally during the analysis of the Configurationally Biased Monte Carlo (CBMC) simulations of 2-ethoxyethanethiol and dodecanethiol decorated ultrasmall Au nanoparticles. 

The directory "figures" contains the analysis script related to each figure reported in the manuscript along with the corresponding data files for clarity. Note that many of the analysis scripts are written to parse xyz trajectory files as input for the calculations, as was done during the analysis, rather than to read in and plot the data presented in text files (provided in the "data" folder). The data files provided were produced during the analysis. 

the directory "example_CBMC_inputs contains" representative LAMMPS/force-field parameters used in the study. Additionally, an example data file for an initially random 50/50 2-ethoxyethanethiol/dodecanethiol configuration is provided that can be visualized via Ovito.

More information on the CBMC model can be found at https://github.com/zlafa24/np-mc.git.







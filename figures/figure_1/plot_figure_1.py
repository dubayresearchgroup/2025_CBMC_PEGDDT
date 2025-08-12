import numpy as np
from scipy.spatial import KDTree
import scipy.stats as ss
import matplotlib.pyplot as plt


def get_binomial(fraction, fragment_size):
    """
    Description: returns binomial distribution for a given surface fraction and fragment size
    Inputs:
    - fraction : float : the ratio of type 1 / (1+2) (or type 2 / (1+2) if type 2 is used as the characterization ligand)
    - fragment size : int : the number of ligands within the desired MALDI-MS fragment family to consider - (e.g. if 4 then the fragments within the family are 0,1,2,3,4)
    """
    return ss.binom.pmf(np.arange(fragment_size + 1), fragment_size, fraction)

def get_maldi_ssr(filled_surface_sites, num_iterations, nn_threshold, target_fractions, fragment_size):
    """
    Input:
    - filled_surface_sites : 
    """
    fragments = [0] * (fragment_size + 1)
    filled_surface_sites_np = [[entry[0], np.array(entry[1])] for entry in filled_surface_sites]
    coordinates_np = np.array([entry[1] for entry in filled_surface_sites_np])
    kdtree = KDTree(coordinates_np)
    total_fragments = 0
    for i in range(num_iterations):
        chosen_index = np.random.randint(0,len(filled_surface_sites_np)-1)
        query_point = filled_surface_sites[chosen_index][1]
        distances, neighbor_indices = kdtree.query(query_point, k=fragment_size, distance_upper_bound=nn_threshold, p=2)
        valid_pairs = [(d, i) for d, i in zip(distances, neighbor_indices) if i != chosen_index and d <= nn_threshold]
        valid_pairs.sort()
        valid_indices = [i for d, i in valid_pairs]
        neighbors = []
        for index in valid_indices:
            if len(neighbors) < 3:
                neighbors.append(index)
        if len(neighbors) == fragment_size - 1:
            fragment_index = int(filled_surface_sites[chosen_index][0] == 1) + sum(1 for index in neighbors if filled_surface_sites[index][0] == 1)
            fragments[fragment_index] += 1
            total_fragments +=1
    if total_fragments == 0:
        raise ValueError("No valid fragments found.")
    fragment_probabilities = [fragment / total_fragments for fragment in fragments]
    ssr = sum((fragment_probabilities[i] - target_fractions[i]) ** 2 for i in range(len(target_fractions)))


    return fragment_probabilities, ssr, total_fragments


if __name__=="__main__":
    study = '2EE_ddt'
    confs = ['janus', 'random', 'striped'] # Initial Configuration Types
    fractions = [10,20,30,40,50,60,70,80,90] #Fraction PEG 
    trials = 6 #Number of trials run for each initial configuration type within each fraction
    anchor_type = 5 #atom type of sulfur anchor
    num_configs=10 #number of snapshots to consider from the final end of the trajectories. These are spaced 10,000 steps apart typically.
    size = 6 #NP core size (icosahedral NP cores are constructed layer wise -- 6 corresponds to the number of layers)
    num_np = 776 #number of np atoms (only the outer NP atoms in the outer 3 layers are considered for computational efficiency -- NP is kept fixed and riged throughout CBMC simulation)
    r_max= 20 #maximum distance for g(r)
    nbins = 60 #number of bins for g(r)
    lig_name_1 = 'ddt' 
    lig_name_2 = '2ethoxyE'
    num_ligands_1 = [111, 102,92, 82, 70, 58, 45, 30, 15] #The number of ligand 1 at each surface fraction PEG listed in fractions
    num_ligands_2 = [12, 25, 39, 54, 70, 87, 105, 123, 143] #The number of ligand 2 at each surface fraction PEG listed in fractions
    num_ligands_1_len = 13 #The length of ligand 1 (DDT has 12 carbons plus 1 sulfur = 13)
    num_ligands_2_len = 6 #The length of ligand 2 (2EE has 1 sulfur, 1 oxygen, and 4 carbons = 6)

    average_frag_dist_per_sf = [] #list to store average fragment distributions
    std_frag_dist_per_sf = [] #list to store standard deviation for fragment distributions
    binomials = [] #list to store the binomials computed at each surface fraction
    plt.figure(figsize=(7,5))
    avg_ssrs = [] #list to store average ssrs
    std_ssrs = [] #list to store standard deviation of ssrs
    for sf, surface_fraction in enumerate(fractions): #iterates through each of the peg surface fractions
        fragments_per_fraction = [] #list to store fragments per surface fraction peg
        ssrs = [] #list to store ssrs per surface fraction peg
        for conf_type in confs: #iterates through each of the initial configuration types (Janus, random, and striped)
            conf_ssr = [] #list to store ssrs per conf type
            for trial in range(trials): #iterates through all 6 trials per conf    
                filepath = f'./{study}/results_5/{surface_fraction}/{conf_type}/t{trial+1}/trajectory.xyz'
                #reads in the trajectory file and parses it out so that we can access the final configurations based on the number of ligands and their length
                data = []
                with open(filepath, 'r') as file:
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1': #excludes nanoparticle atoms
                                data.append(columns)
                total_lines = len(data)
                data = np.array(data[(total_lines - num_configs * (num_ligands_1[sf] * num_ligands_1_len + num_ligands_2[sf] * num_ligands_2_len)):], dtype=float)
                for config in range(num_configs): #iterates through the number of configurations read in from the end to consider
                    print(surface_fraction, conf_type, trial+1, config+1)
                    start_line = config * (num_ligands_1[sf] * num_ligands_1_len + num_ligands_2[sf] * num_ligands_2_len)
                    end_line = start_line + (num_ligands_1[sf] * num_ligands_1_len + num_ligands_2[sf] * num_ligands_2_len)
                    lig_sites = data[start_line:end_line, 1:4]
                    type_sites = data[start_line:end_line, 0]
                    #parse out the anchors from each configuration and assign them a type 1 or 2 depending on ligand ID
                    lig_1_sites = []
                    lig_2_sites = []
                    count = 0
                    for k in range(len(type_sites)):
                        if data[k][0] == anchor_type:
                            if count < num_ligands_1[sf]:
                                lig_1_sites.append(data[k][1:4])
                            else:
                                lig_2_sites.append(data[k][1:4])
                            count +=1
                    lig_1_sites = np.array(lig_1_sites)
                    lig_2_sites = np.array(lig_2_sites)
                    L1_sites_occupied = {tuple(site) for site in lig_1_sites}
                    L2_sites_occupied = {tuple(site) for site in lig_2_sites}
                    filled_sites = [[1, list(site)] for site in L1_sites_occupied] + [[2, list(site)] for site in L2_sites_occupied]
                    #compute binomial
                    target_fractions = get_binomial(float(num_ligands_1[sf]) / (float(num_ligands_1[sf]) + float(num_ligands_2[sf])), 4)                    
                    #compute MALDI-MS distribution and SSR using binomial (target_fractions)
                    fragment_probabilities, ssr, total_fragments = get_maldi_ssr(filled_sites, 50000, 6, target_fractions, 4)
                    ssrs.append(ssr)
                    conf_ssr.append(ssr)
            if conf_type == 'janus':
                if surface_fraction == 10: #include label in legend -- ensures single legend entry
                    plt.errorbar((surface_fraction/100), y=np.mean(conf_ssr), yerr=np.std(conf_ssr),  fmt='o', label=f'Janus', color='k', ecolor='k', capsize=5)
                else:
                    plt.errorbar((surface_fraction/100), y=np.mean(conf_ssr), yerr=np.std(conf_ssr), fmt='o', color='k', ecolor='k', capsize=5)
            elif conf_type == 'random':
                if surface_fraction == 10:#include label in legend -- ensures single legend entry
                    plt.errorbar((surface_fraction/100), y=np.mean(conf_ssr), yerr=np.std(conf_ssr), fmt='o', label=f'Random', color='#3d3a3a', ecolor='#3d3a3a', capsize=5)
                else:
                    plt.errorbar((surface_fraction/100), y=np.mean(conf_ssr), yerr=np.std(conf_ssr), fmt='o', color='#3d3a3a', ecolor='#3d3a3a', capsize=5)
            elif conf_type == 'striped':
                if surface_fraction == 10: #include label in legend -- ensures single legend entry
                    plt.errorbar((surface_fraction/100), y=np.mean(conf_ssr), yerr=np.std(conf_ssr), fmt='o', label=f'Striped', color='#7b7474', ecolor='#7b7474', capsize=5)
                else:
                    plt.errorbar((surface_fraction/100), y=np.mean(conf_ssr), yerr=np.std(conf_ssr), fmt='o', color='#7b7474', ecolor='#7b7474', capsize=5)

ssrs_from_janus_configs = np.genfromtxt('./janus_fragment_distributions.txt', skip_header=1, usecols=1, dtype=float)
ssrs_from_janus_configs_std = np.genfromtxt('./janus_fragment_distributions.txt', skip_header=1, usecols=2, dtype=float)
plt.fill_between([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], ssrs_from_janus_configs, color='k', alpha=0.1, interpolate=True)
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], ssrs_from_janus_configs+ssrs_from_janus_configs_std, color='k', alpha=0.8, linestyle=':')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], ssrs_from_janus_configs, color='k', alpha=0.8, label='SSR Upper Bound')
plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], ssrs_from_janus_configs-ssrs_from_janus_configs_std, color='k', alpha=0.8, linestyle=':')
plt.xlabel(r'Surface Fraction PEG ($x_{\mathrm{PEG}}$)')
plt.ylabel('Sum Squared Resdiual (SSR)')
plt.ylim(0,0.4)
plt.xlim(0,1)
plt.xticks([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90], ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
plt.legend(loc="upper center", ncols = 5, fontsize='medium', frameon=False)
plt.tight_layout()
plt.savefig(f'figure_1.png', format='png', dpi=600)
plt.close()
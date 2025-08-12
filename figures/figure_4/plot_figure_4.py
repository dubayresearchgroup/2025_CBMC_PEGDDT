import numpy as np
from scipy.spatial import KDTree
import scipy.stats as ss
import matplotlib.pyplot as plt
import math

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
    experimental_surface_fractions = np.genfromtxt('./experimental_ssrs.txt', usecols=8, skip_header=2)
    experimental_ssrs = np.genfromtxt('./experimental_ssrs.txt', usecols=2, skip_header=2)
    
    with open(f'{study}_surface_fraction_data.txt', 'w') as file:
        file.write('##Surface Fraction data 2.8 nm Au NP with surface area of 25.949 nm2 and 17.31 angstrom edge length\n# #DDT\t#PEG\tPEG (%)\tInitial_Config\ttrial\tbinomial_probabilities\tfragment_probabilities\tssr\n')
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
                    fragments_per_fraction.append(fragment_probabilities)
                    if trial == 0:
                        binomials.append(target_fractions)
                    print(f'{ssr:.3f}')
        print(surface_fraction, np.mean(ssrs))
        avg_ssrs.append(np.mean(ssrs))
        std_ssrs.append(np.std(ssrs))
        with open(f'{study}_ssr_peg_surface_fraction.txt', 'a') as data_file:
            data_file.write(f"{surface_fraction}\t{avg_ssrs}\t{std_ssrs}\n")
        plt.errorbar(surface_fraction/100, y=np.mean(ssrs), yerr=np.std(ssrs),  fmt='o', capsize=5, color='#E57200', ecolor='k')
        print(f'{surface_fraction}SSR: ',np.mean(ssrs), '+/- ',  np.std(ssrs))
        average_frag_dist_per_sf.append([sum(values) / len(values) for values in zip(*fragments_per_fraction)])
        std_frag_dist_per_sf.append([math.sqrt(sum((x - (sum(values) / len(values))) ** 2 for x in values) / len(values)) for values in zip(*fragments_per_fraction)])
    plt.errorbar(-5, 0, yerr=np.std(ssrs), fmt='o', capsize=5, color='#E57200', ecolor='k', label='Simulation')
    plt.scatter(experimental_surface_fractions, experimental_ssrs,   color='#232D4B', marker='*', label='Experiment')
    plt.xlabel(r'Surface Fraction PEG ($x_{\mathrm{PEG}}$)')
    plt.ylabel('Sum Squared Resdiual (SSR)')
    plt.ylim(0,0.4)
    plt.xlim(0,1)
    plt.xticks([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90], ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
    plt.legend(loc="upper center", ncols = 5, fontsize='medium', frameon=False)
    plt.tight_layout()
    plt.savefig(f'figure_5.png', format='png', dpi=600)
    plt.close()

    #if there are multiple experimental data points within 0.01 fraction PEG from simulation the average frag distribution is reported with std. Else the nearest data point is used as reference
    #10: average of 0.0925951723142031, 0.10716685004118605
    #20: average of 0.21058049335838974, 0.21072060171800328, 0.19488481105193506
    #30: single point at 0.33455966422710004
    #40: single point at 0.35454313385320196
    #50: single point at 0.4796256804107461 
    #60: single point at 0.6220057359860099
    #70: single point at 0.7043771360215466
    #80: average of 0.8078004681891742, 0.8042197041189715
    #90: single point at 0.9386814387865471
    exp_frag_dists = [[0.01806944, 0.01873145, 0.03163181, 0.19067202, 0.74089528],
                    [0.06732846, 0.06040886, 0.09469415, 0.18735002, 0.59021851],
                    [0.1252174905439155, 0.08575697134022946, 0.18642214046849298, 0.2072534997750636, 0.3953498978722984],
                    [0.20380003987630915, 0.06981628145750698, 0.09457936618029911, 0.2043647991744521, 0.4274395133114327],
                    [0.26288, 0.10305, 0.14289, 0.20413, 0.28706],
                    [0.4080880559195217, 0.1405915553331039, 0.1570648279357051, 0.11976639839523125, 0.17448916241643803],
                    [0.51287, 0.14165, 0.15994, 0.05342, 0.13213],
                    [0.6253509713992805, 0.18311365023960732, 0.06488127563199302, 0.04353295703636094, 0.08312114569275814],
                    [0.8692491638427817, 0.07809401049676672, 0.020240871046377014, 0.002965326192007086, 0.029450628422067263]]
    exp_std_frag_dists = [[0.00320211, 0.00225224, 0.01470153, 0.01678472, 0.01858818],
                        [0.00768967, 0.04052796, 0.04146459, 0.03539329, 0.0242477],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0.08096176563854296, 0.08843882644533654, 0.018462405002508748, 0.014444245072739444, 0.04038371088204177],
                        [0,0,0,0,0]
    ]

    all_frac_ssrs = []
    for i, ii in enumerate(exp_frag_dists):
        ssr = 0
        for j, jj in enumerate(ii):
            ssr += (jj - average_frag_dist_per_sf[i][j])**2
        print(fractions[i],': ',ssr)
    print('Average SSR across all PEG fractions: ', np.mean(all_frac_ssrs), '+/-', np.std(all_frac_ssrs))
    fig, axs = plt.subplots(1, 9, figsize=(7, 4))
    for i, frag_dist in enumerate(average_frag_dist_per_sf):
        if i == 7:
            axs[i].bar([-0.15, 0.85, 1.85, 2.85, 3.85], frag_dist, yerr=std_frag_dist_per_sf[i], width = 0.3, color='#E57200', label='Simulation',capsize=1, ecolor='k', error_kw={'elinewidth': 1, 'capthick': 1})
            axs[i].bar([0.15, 1.15, 2.15, 3.15, 4.15], exp_frag_dists[i], yerr=exp_std_frag_dists[i], color="#56B4E9", width = 0.3, label='Experiment',  capsize=1, ecolor='k', error_kw={'elinewidth': 1, 'capthick': 1})
        else:
            axs[i].bar([-0.15, 0.85, 1.85, 2.85, 3.85], frag_dist, yerr=std_frag_dist_per_sf[i], width = 0.3, color='#E57200', capsize=1, ecolor='k', error_kw={'elinewidth': 1, 'capthick': 1})
            axs[i].bar([0.15, 1.15, 2.15, 3.15, 4.15], exp_frag_dists[i], yerr=exp_std_frag_dists[i], width = 0.3, color='#56B4E9', capsize=1, ecolor='k', error_kw={'elinewidth': 1, 'capthick': 1})
        axs[i].set_xlim(-0.5, 4.5)
        axs[i].set_xticks([0, 1, 2, 3, 4])
        axs[i].set_xticklabels([0, 1, 2, 3, 4], fontsize='medium')
        axs[i].set_ylim(0, 1.0)
        if i != 0:
            axs[i].set_yticks([0.2,0.4,0.6,0.8,1.0],['','','','',''])
        axs[i].set_title(f'{fractions[i]/100}', fontsize='medium')
    axs[4].set_xlabel('# DDT in Fragment')
    axs[0].set_ylabel('Fraction')
    fig.suptitle("Surface Fraction PEG ($x_{\mathrm{PEG}}$)", fontsize=9.5)
    fig.savefig('figure_4.png', format='png', dpi=600)
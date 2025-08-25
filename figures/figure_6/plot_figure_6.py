import numpy as np
from scipy.spatial import KDTree
import scipy.stats as ss
import matplotlib.pyplot as plt
import math

def propagate_division(a, sigma_a, b, sigma_b):
    """
    Standard error propagation for division
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")

    result = a / b
    rel_error_squared = (sigma_a / a) ** 2 + (sigma_b / b) ** 2
    sigma_result = result * math.sqrt(rel_error_squared)

    return result, sigma_result

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
    - filled_surface_sites : array : coordinates of sulfur anchor positions typed as 1 or 2 depending on ligand identity
    - num_iterations : int : defines the number of MALDI fragmentation samples to run (typical is 50,000x but for larger, more dense, monolayers a larger value will result in better sampling)
    - nn_threshold : float/int : defines the nearest neighbor threshold that the function will consider part of the same fragment (default is 6, though this value should depend on the first solvation shell of the anchor-anchor g(r)).
    - target_fractions : list of floats : the binomial distribution to compare the obtained MALDI distribution to in order to obtain the SSR value
    - fragment_size : int : The maximum number of 1 type of ligand present in the desired fragment family (i.e. Au4L4 would mean fragment_size = 4). Ideally, multiple fragment families should be considered.
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

    ssrs_from_janus_configs = np.genfromtxt('./janus_fragment_distributions.txt', skip_header=1, usecols=1, dtype=float)
ssrs_from_janus_configs_std = np.genfromtxt('./janus_fragment_distributions.txt', skip_header=1, usecols=2, dtype=float)
avg_ssrs = [0.022111111111111113, 0.07744444444444445, 0.12455555555555556, 0.14922222222222226, 0.19172222222222224, 0.22927777777777777, 0.21605555555555558, 0.1831111111111111, 0.08272222222222222]
std_ssrs = [0.013943891091278363, 0.027522157179306804, 0.0358069239459233, 0.038810874128487285, 0.050206026149541005, 0.06141914789701223, 0.038108721985367404, 0.03309227652235637, 0.011473668189746253]

janus_character = []
janus_character_error = []
for i in range(len(fractions)):
    a,b = propagate_division(avg_ssrs[i], std_ssrs[i], ssrs_from_janus_configs[i], ssrs_from_janus_configs_std[i])
    janus_character.append(a)
    janus_character_error.append(b)
print(janus_character)
plt.figure(figsize=(8, 3.5))
plt.errorbar(x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], y=janus_character,  yerr=janus_character_error,  color='k',  capsize=5,  marker='o',  linestyle='-')
plt.xlabel(r'Surface Fraction PEG ($x_{\mathrm{PEG}}$)')
plt.ylabel(r'$\overline{\mathrm{SSR}}_{\mathrm{sim}} \,/\, \overline{\mathrm{SSR}}_{\mathrm{Janus}}$')
plt.ylim(0.0, 1.2)
plt.xlim(0.05, 0.95)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize='small')
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#plt.axvline(x=0, color='k', linewidth=3.5)
plt.axvline(x=0.1, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.2, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.3, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.4, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.5, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.6, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.7, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.8, color='k', linestyle=':', alpha=0.15)
plt.axvline(x=0.9, color='k', linestyle=':', alpha=0.15)
#plt.legend(loc='upper center', frameon=False, fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()

plt.savefig('janus_character.png', format='png', dpi=600)

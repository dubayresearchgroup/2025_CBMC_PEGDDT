import numpy as np
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_clusters(filled_surface_sites, nn_threshold):
    filled_surface_sites_np = [[entry[0], np.array(entry[1])] for entry in filled_surface_sites]
    coordinates_np = np.array([entry[1] for entry in filled_surface_sites_np])
    kdtree = KDTree(coordinates_np)

    visited_indices = set()
    clusters = []

    def dfs(index, cluster):
        if index in visited_indices:
            return
        visited_indices.add(index)
        cluster.append(index)

        query_point = filled_surface_sites[index][1]
        distances, neighbor_indices = kdtree.query(query_point, k=10, distance_upper_bound=nn_threshold, p=2)

        for neighbor_index in neighbor_indices:
            if 0 <= neighbor_index < len(filled_surface_sites):
                if filled_surface_sites[neighbor_index][0] == filled_surface_sites[index][0]:
                    dfs(neighbor_index, cluster)

    for chosen_index in range(len(filled_surface_sites)):
        if chosen_index not in visited_indices:
            new_cluster = []
            dfs(chosen_index, new_cluster)
            if new_cluster:
                clusters.append(new_cluster)

    return clusters


def compute_cluster_area(cluster_coords, np_coords, used_surface_atoms, file, cluster_id, au_threshold=2.6):
    kdtree = KDTree(np_coords)
    nearby_au_indices = set()
    for coord in cluster_coords:
        indices = kdtree.query_ball_point(coord, r=au_threshold)
        for idx in indices:
            if idx not in used_surface_atoms:
                nearby_au_indices.add(idx)
    
    used_surface_atoms.update(nearby_au_indices)
    for idx in nearby_au_indices:
        file.write(f'{cluster_id+2}\t{np_coords[idx][0]}\t{np_coords[idx][1]}\t{np_coords[idx][2]}\n')
    return len(nearby_au_indices), nearby_au_indices


    
def analyze_clusters(filled_surface_sites, np_coords, nn_threshold, lig_1_type, lig_2_type, num_ligands_1, num_ligands_2, num_surface_atoms, file):
    type1_clusters = []
    type2_clusters = []
    type1_areas = []
    type2_areas = []
    cluster_id = 0
    ligand_types = set(entry[0] for entry in filled_surface_sites)
    used_surface_atoms = set()
    au_atoms_covered = set()
    file.write(f'{len(np_coords)}\n\n')
    for ligand_type in ligand_types:
        ligand_sites = [entry for entry in filled_surface_sites if entry[0] == ligand_type]
        ligand_clusters = get_clusters(ligand_sites, nn_threshold)
        # Filter out single-point clusters
        ligand_clusters = [cluster for cluster in ligand_clusters if len(cluster) > 0]
        random.shuffle(ligand_clusters)
        cluster_fractions = []
        cluster_areas = []
        for cluster in ligand_clusters:
            coords = np.array([ligand_sites[i][1] for i in cluster])
            fraction = len(cluster) #number of ligands
            cluster_fractions.append(fraction)
                
            num_surface_atoms_spanned_by_cluster, au_indices = compute_cluster_area(coords, np_coords, used_surface_atoms, file, cluster_id)
            au_atoms_covered.update(au_indices)
            area = (num_surface_atoms_spanned_by_cluster/num_surface_atoms)#*np_surface_area
            cluster_areas.append(area)

        if ligand_type == lig_1_type:
            type1_clusters.extend(cluster_fractions)
            type1_areas.extend(cluster_areas)
        elif ligand_type == lig_2_type:
            type2_clusters.extend(cluster_fractions)
            type2_areas.extend(cluster_areas)
        cluster_id += 1

    for i, coord in enumerate(np_coords):
        if i not in au_atoms_covered:
            file.write(f'1\t{coord[0]}\t{coord[1]}\t{coord[2]}\n')
    
    return type1_clusters, type2_clusters, np.array(type1_areas), np.array(type2_areas)


study = 'peg6p29_weighted_trials_3day_runs'
confs = ['janus', 'random', 'striped'] #'janus', , 'striped'
fractions = [10,20,30,40,50,60,70,80,90] #Fraction PEG 
trials = 6
anchor_type = 5
num_configs=10
size = 6
num_np = 776
r_max= 20
nbins = 60
lig_name_1 = 'ddt'
lig_name_2 = '2ethoxyE'
num_ligands_1 = [111, 102,92, 82, 70, 58, 45, 30, 15]#[111, 102, 92, 82, 70, 58, 45]  #[119, 114, 106, 97, 85, 72] 
num_ligands_2 = [12, 25, 39, 54, 70, 87, 105, 123, 143]#[12, 25, 39, 54, 70, 87, 105]  #[13, 28, 46, 65, 85, 108] 
num_ligands_1_len = 13
num_ligands_2_len = 6
num_np_atoms = 776

np_coords = []
with open('np_6h.xyz', 'r') as file:
    for line in file:
        if line.strip():
            columns = line.split()
            if len(columns) == 4:
                np_coords.append([float(c) for c in columns[1:4]])
np_coords = np.array(np_coords)
print(np_coords)
nw_10_1, nw_10_2 = [], []
nw_20_1, nw_20_2 = [], [] 
nw_30_1, nw_30_2 = [], [] 
nw_40_1, nw_40_2 = [], [] 
nw_50_1, nw_50_2 = [], [] 
nw_60_1, nw_60_2 = [], [] 
nw_70_1, nw_70_2 = [], [] 
nw_80_1, nw_80_2 = [], [] 
nw_90_1, nw_90_2 = [], []
nw_1_lists = [nw_10_1, nw_20_1, nw_30_1, nw_40_1, nw_50_1, nw_60_1, nw_70_1, nw_80_1, nw_90_1]
nw_2_lists = [nw_10_2, nw_20_2, nw_30_2, nw_40_2, nw_50_2, nw_60_2, nw_70_2, nw_80_2, nw_90_2]
ss_area = np.pi * (2.88499/2)**2 #radial area per Au atom but this excludes the roughly triangular portion that exists between every set of 3 Au atoms
num_surface_atoms = 362 # number of Au atoms
surface_area = 25.949 #nm2
sa_10_1, sa_10_2 = [], []
sa_20_1, sa_20_2 = [], [] 
sa_30_1, sa_30_2 = [], [] 
sa_40_1, sa_40_2 = [], [] 
sa_50_1, sa_50_2 = [], [] 
sa_60_1, sa_60_2 = [], [] 
sa_70_1, sa_70_2 = [], [] 
sa_80_1, sa_80_2 = [], [] 
sa_90_1, sa_90_2 = [], []
total_fractions_avg = []
total_fractions_std = []
sa_1_lists = [sa_10_1, sa_20_1, sa_30_1, sa_40_1, sa_50_1, sa_60_1, sa_70_1, sa_80_1, sa_90_1]
sa_2_lists = [sa_10_2, sa_20_2, sa_30_2, sa_40_2, sa_50_2, sa_60_2, sa_70_2, sa_80_2, sa_90_2]  
peg_colors = ['#ff9898','#fc8987','#f87a76','#f46b65','#ee5c53','#e84c41','#e13a2e','#da251a','#d10000']
alkane_colors = ['#6d6d6d','#5f5f5f','#515151','#444444','#373737','#2a2a2a','#1e1e1e','#131313','#000000']
inter_colors = ['#8095ff','#7285f6','#6576ec','#5867e2','#4a57d8','#3d48ce','#2e39c3','#1d29b8','#0017ad']
oe_colors = ['#ffe070','#f8d664','#f1cc58','#eac14d','#e3b741','#dcad34','#d6a327','#cf9917','#c88f00']
fig, axs = plt.subplots(1,3, sharex=True,sharey=True)
bounds = list(range(len(fractions) + 1))  # Create bounds based on the length of fractions
norm = mpl.colors.BoundaryNorm(bounds, len(fractions))
cmap_pegs = mpl.colors.ListedColormap(peg_colors[:len(fractions)])
cmap_alkanes = mpl.colors.ListedColormap(alkane_colors[:len(fractions)])
cmap_inters = mpl.colors.ListedColormap(inter_colors[:len(fractions)])
cmap_ethers = mpl.colors.ListedColormap(oe_colors[:len(fractions)])
with open('np_surface_area_clusters.xyz', 'w') as xyz_file:
    for i, fraction in enumerate(fractions):
        total_trials = len(confs) * trials
        total_fractions = []
        for j, conf_type in enumerate(confs):
            for trial in range(trials):
                if fraction <= 20:
                    filepath = f'./{study}/results_4/{fraction}/{conf_type}/t{trial+1}/trajectory.xyz'
                    if filepath == f'./{study}/results_4/10/janus/t5/trajectory.xyz': #particular file was not reinitialized correctly.
                        filepath = f'./{study}/results_3/10/janus/t5/trajectory.xyz'
                else:
                    filepath = f'./{study}/results_5/{fraction}/{conf_type}/t{trial+1}/trajectory.xyz'
                data = []
                with open(filepath, 'r') as file:
                    for line in file:
                        if line.strip():
                            columns = line.split()
                            if len(columns) == 4 and columns[0] != '1':
                                data.append(columns)
                total_lines = len(data)
                data = np.array(data[(total_lines - num_configs * (num_ligands_1[i] * num_ligands_1_len + num_ligands_2[i] * num_ligands_2_len)):], dtype=float)
                for config in range(num_configs):
                    start_line = config * (num_ligands_1[i] * num_ligands_1_len + num_ligands_2[i] * num_ligands_2_len)
                    end_line = start_line + (num_ligands_1[i] * num_ligands_1_len + num_ligands_2[i] * num_ligands_2_len)
                    lig_sites = data[start_line:end_line, 1:4]
                    type_sites = data[start_line:end_line, 0]
                    oe_sites = []
                    ch3_sites = []
                    peg_ch3_sites = []
                    ch3_s_sites = []
                    peg_s_sites = [] 
                    count_s = 0
                    for k in range(len(type_sites)):
                        if type_sites[k] == anchor_type:
                            count_s += 1
                            if count_s <= num_ligands_1[i]:
                                ch3_s_sites.append(lig_sites[k])
                            else:
                                peg_s_sites.append(lig_sites[k])
                        elif type_sites[k] == 4: #CH3
                            if count_s <= num_ligands_1[i]:
                                ch3_sites.append(lig_sites[k])
                            else:
                                peg_ch3_sites.append(lig_sites[k])
                        elif type_sites[k] == 6: #H-bearing oxygen
                            oe_sites.append(lig_sites[k])

                    oe_sites = np.array(oe_sites)
                    ch3_sites = np.array(ch3_sites)
                    peg_ch3_sites = np.array(peg_ch3_sites)
                    oe_s_sites = np.array(peg_s_sites)
                    ch3_s_sites = np.array(ch3_s_sites)
                    oe_sites_occupied = {tuple(site) for site in oe_sites}
                    ch3_sites_occupied = {tuple(site) for site in ch3_sites}
                    peg_ch3_sites_occupied = {tuple(site) for site in peg_ch3_sites}
                    ch3_s_sites_occupied = {tuple(site) for site in ch3_s_sites}
                    oe_s_sites_occupied = {tuple(site) for site in oe_s_sites}
                    filled_s_sites = [['peg', list(site)] for site in oe_s_sites_occupied] + [['ch3', list(site)] for site in ch3_s_sites_occupied]
                    head_group_sites = [['peg', list(site)] for site in peg_ch3_sites_occupied] + [['ch3', list(site)] for site in ch3_sites_occupied]
                    oe_sites = [['peg', list(site)] for site in peg_ch3_sites_occupied]
                    aa, bb, aa_area, bb_area = analyze_clusters(
                        filled_surface_sites=filled_s_sites,
                        np_coords=np_coords,
                        nn_threshold=6,
                        lig_1_type='ch3',
                        lig_2_type='peg',
                        num_ligands_1=num_ligands_1[i],
                        num_ligands_2=num_ligands_2[i],
                        num_surface_atoms=num_surface_atoms,
                        file = xyz_file
                    )
                    print(aa_area, bb_area)
                    if aa and sum(aa) > 0:
                        aa_np = np.array(aa)
                        nw_1_lists[i].append(np.sum(aa_np ** 2) / np.sum(aa_np))
                        sa_1_lists[i].append((np.sum(aa_area ** 2) / np.sum(aa_area)))
                        print('CH3:', nw_1_lists[i][-1])
                        print('FRACTION CH3 ligands consumed:', sa_1_lists[i][-1])
                    else:
                        nw_1_lists[i].append(0)
                        sa_1_lists[i].append(0)
                    if bb and sum(bb) > 0:
                        bb_np = np.array(bb)
                        nw_2_lists[i].append(np.sum(bb_np ** 2) / np.sum(bb_np))
                        sa_2_lists[i].append((np.sum(bb_area ** 2) / np.sum(bb_area)))
                        print('FRACTION PEG ligands consumed:', sa_2_lists[i][-1])
                    else:
                        nw_2_lists[i].append(0)
                        sa_2_lists[i].append(0)
                    total_fraction = 0
                    if aa and sum(aa) > 0:
                        total_fraction += np.sum(aa_area)
                    if bb and sum(bb) > 0:
                        total_fraction += np.sum(bb_area) 
                    print('FRACTION OF TOTAL SPACE CONSUMED BY LIGANDS:', total_fraction)
                    if total_fraction > 1.0:
                        print("ERROR: FRACTION OF TOTAL SPACE CONSUMED BY BOTH LIGANDS EXCEEDS TOTAL NP SURFACE AREA")
                    total_fractions.append(total_fraction)
        total_fractions_avg.append(np.mean(total_fractions))
        total_fractions_std.append(np.std(total_fractions))
    xyz_file.close()
print(total_fractions_std)
print(len(nw_30_1), len(nw_30_2))
print(len(sa_30_1), len(sa_30_2))

ligand_area_fraction = ss_area / 100 #3531.97
ddt_fraction = np.array(num_ligands_1) * ligand_area_fraction
peg_fraction = np.array(num_ligands_2) * ligand_area_fraction
total_ligands = np.array(num_ligands_1) + np.array(num_ligands_2)
fig, ax1 = plt.subplots()
#ax2.plot([0.08,0.18,0.28,0.38,0.48,0.58,0.68,0.78,0.88], np.array(num_ligands_1), color='#232D4B', alpha=0.5)
#ax2.plot([0.12,0.22,0.32,0.42,0.52,0.62,0.72,0.82,0.92], np.array(num_ligands_2), color='#F84C1E', alpha=0.5)
box_1a = ax1.boxplot(sa_10_1, positions=[0.08], vert=True, patch_artist=True, widths=0.0325)
box_1b = ax1.boxplot(sa_10_2, positions=[0.12], vert=True, patch_artist=True, widths=0.0325)
box_2a = ax1.boxplot(sa_20_1, positions=[0.18], vert=True, patch_artist=True, widths=0.0325)
box_2b = ax1.boxplot(sa_20_2, positions=[0.22], vert=True, patch_artist=True, widths=0.0325)
box_3a = ax1.boxplot(sa_30_1, positions=[0.28], vert=True, patch_artist=True, widths=0.0325)
box_3b = ax1.boxplot(sa_30_2, positions=[0.32], vert=True, patch_artist=True, widths=0.0325)
box_4a = ax1.boxplot(sa_40_1, positions=[0.38], vert=True, patch_artist=True, widths=0.0325)
box_4b = ax1.boxplot(sa_40_2, positions=[0.42], vert=True, patch_artist=True, widths=0.0325)
box_5a = ax1.boxplot(sa_50_1, positions=[0.48], vert=True, patch_artist=True, widths=0.0325)
box_5b = ax1.boxplot(sa_50_2, positions=[0.52], vert=True, patch_artist=True, widths=0.0325)
box_6a = ax1.boxplot(sa_60_1, positions=[0.58], vert=True, patch_artist=True, widths=0.0325)
box_6b = ax1.boxplot(sa_60_2, positions=[0.62], vert=True, patch_artist=True, widths=0.0325)
box_7a = ax1.boxplot(sa_70_1, positions=[0.68], vert=True, patch_artist=True, widths=0.0325)
box_7b = ax1.boxplot(sa_70_2, positions=[0.72], vert=True, patch_artist=True, widths=0.0325)
box_8a = ax1.boxplot(sa_80_1, positions=[0.78], vert=True, patch_artist=True, widths=0.0325)
box_8b = ax1.boxplot(sa_80_2, positions=[0.82], vert=True, patch_artist=True, widths=0.0325)
box_9a = ax1.boxplot(sa_90_1, positions=[0.88], vert=True, patch_artist=True, widths=0.0325)
box_9b = ax1.boxplot(sa_90_2, positions=[0.92], vert=True, patch_artist=True, widths=0.0325)
boxes = [box_1a,box_1b,box_2a,box_2b,box_3a,box_3b,box_4a,box_4b,box_5a,box_5b,box_6a,box_6b,box_7a,box_7b,box_8a,box_8b,box_9a,box_9b]
for b,box in enumerate(boxes):
    if b %2 == 0:
        box['boxes'][0].set_facecolor('#232D4B')
        box['medians'][0].set_color('gray')
        box['medians'][0].set_linewidth(2)
    else:
        box['boxes'][0].set_facecolor('#F84C1E')
        box['medians'][0].set_color('gray')
        box['medians'][0].set_linewidth(2)
ax1.scatter([], [], label='DDT', color=box_1a['boxes'][0].get_facecolor(), marker='s')
ax1.scatter([], [], label='PEG', color=box_1b['boxes'][0].get_facecolor(), marker='s')
ax1.errorbar([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], total_fractions_avg, yerr=total_fractions_std, color='k', label='Total$_{DDT+PEG}$')
ss_area_nm2 = ss_area / 100
ax1.set_ylabel('Number Weighted Average Patch Surface Fraction')
ax1.set_xlabel(r'$x_{\mathrm{PEG}}$')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.1)
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.xticks([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90], ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0.1, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.20, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.30, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.40, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.50, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.60, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.70, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.80, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.axvline(x=0.90, ymax = 0.90909, alpha=0.1, linestyle=':', color='k')
ax1.legend(loc='upper center', ncols=3, frameon=False, fontsize='large')
plt.tight_layout()
plt.savefig('figure_s1.png', format='png', dpi=600)
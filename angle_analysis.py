#!/usr/bin/env python3

import os
import pickle
import sys

import numpy                as np
import matplotlib.pyplot    as plt

from matplotlib.patches         import Patch
from scipy.sparse               import coo_matrix
from scipy.sparse.csgraph       import connected_components, breadth_first_order
from scipy.spatial              import cKDTree
from sklearn.model_selection    import train_test_split



##################
### PARAMETERS ###
##################

TRAINING_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1
PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
DIR_NAME = "resnet50_2"



#################
### FUNCTIONS ###
#################

def largest_cluster(pos, box, cutoff=1.5 * 2.889):
    n = len(pos)
    pos = np.mod(pos, box)
    pos[pos >= box] = 0.0
    pairs = cKDTree(pos, boxsize=box).query_pairs(cutoff, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    keep = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    order, pred = breadth_first_order(graph, keep[0], directed=False)
    unwrapped = np.zeros_like(pos)
    unwrapped[order[0]] = pos[order[0]]
    for i in order[1:]:
        d = pos[i] - pos[pred[i]]
        unwrapped[i] = unwrapped[pred[i]] + d - box * np.rint(d / box)
    return keep, unwrapped[keep]



############
### PLOT ###
############

def cs_vs_angle(alpha, y_true, err):
    plt.scatter(alpha, y_true[:, 2], c=err[:, 2], cmap='viridis', s=4, edgecolor='none')
    plt.xlabel('Angle between COM axis and z-axis (degrees)', fontsize=13)
    plt.ylabel('True coreshell index', fontsize=13)
    plt.tight_layout()
    plt.colorbar(label='Error on coreshell index prediction')
    plt.savefig("cs_vs_angle.png", dpi=300, bbox_inches='tight')
    plt.close()


def error_vs_angle_hist2d(alpha, err):
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    h0 = ax[0].hist2d(alpha, err[:, 2], bins=90, cmap=cmap)
    ax[0].set_axisbelow(True)
    ax[0].grid(alpha=0.25, ls='-', lw=0.5)
    ax[0].set_xlabel('Angle between COM axis and z-axis (degrees)', fontsize=13)
    ax[0].set_ylabel(r'Error on coreshell index prediction $\left| \hat{y} - y \right|$', fontsize=13)
    fig.colorbar(h0[3], ax=ax[0], label="Number of images")    
    mask = np.where(err[:, 2] < 0.2)
    h1 = ax[1].hist2d(alpha[mask], err[:, 2][mask], bins=45, cmap=cmap)
    ax[1].set_axisbelow(True)
    ax[1].grid(alpha=0.25, ls='-', lw=0.5)
    ax[1].set_xlabel('Angle between COM axis and z-axis (degrees)\n(Zoom error < 0.2)', fontsize=13)
    ax[1].set_ylabel(r'Error on coreshell index prediction $\left| \hat{y} - y \right|$', fontsize=13)
    fig.colorbar(h1[3], ax=ax[1], label="Number of images")
    plt.tight_layout()
    plt.savefig("error_vs_angle_hist2d.png", dpi=300, bbox_inches='tight')
    plt.close()


def error_vs_angle_hist2d_normalized(alpha, err):
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    H0, xedges0, yedges0 = np.histogram2d(alpha, err[:, 2], bins=90)
    col_max0 = H0.max(axis=0 if False else 1, keepdims=True)
    H_norm0 = np.where(col_max0 > 0, H0 / col_max0, np.nan)
    #H_norm0[H0 == 0] = np.nan
    mesh0 = ax[0].pcolormesh(xedges0, yedges0, H_norm0.T, cmap=cmap, vmin=0, vmax=1)
    ax[0].set_axisbelow(True)
    ax[0].grid(alpha=0.25, ls='-', lw=0.5)
    ax[0].set_xlabel('Angle between COM axis and z-axis (degrees)', fontsize=13)
    ax[0].set_ylabel(r'Error on coreshell index prediction $\left| \hat{y} - y \right|$', fontsize=13)
    fig.colorbar(mesh0, ax=ax[0], label="Normalized density per column")
    mask = np.where(err[:, 2] < 0.2)
    H1, xedges1, yedges1 = np.histogram2d(alpha[mask], err[:, 2][mask], bins=45)
    col_max1 = H1.max(axis=0 if False else 1, keepdims=True)
    H_norm1 = np.where(col_max1 > 0, H1 / col_max1, np.nan)
    #H_norm1[H1 == 0] = np.nan
    mesh1 = ax[1].pcolormesh(xedges1, yedges1, H_norm1.T, cmap=cmap, vmin=0, vmax=1)
    ax[1].set_axisbelow(True)
    ax[1].grid(alpha=0.25, ls='-', lw=0.5)
    ax[1].set_xlabel('Angle between COM axis and z-axis (degrees)\n(Zoom error < 0.2)', fontsize=13)
    ax[1].set_ylabel(r'Error on coreshell index prediction $\left| \hat{y} - y \right|$', fontsize=13)
    fig.colorbar(mesh1, ax=ax[1], label="Normalized density per column")
    plt.tight_layout()
    plt.savefig("error_vs_angle_hist2d_norm.png", dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_mean_error_vs_angle_hist(alpha, err):
    bins = np.arange(0, 100, 10)
    bin_idx = np.digitize(alpha, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_err = np.full(len(centers), np.nan)
    std_err = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i_bin in range(len(centers)):
        e = err[bin_idx == i_bin, 2]
        counts[i_bin] = len(e)
        if len(e):
            mean_err[i_bin] = e.mean()
            std_err[i_bin] = e.std() / np.sqrt(len(e))  # erreur standard sur la moyenne
    
    i_best = np.nanargmin(mean_err)  # index of the bin with the lowest mean error
    colors = ["#4C72B0"] * len(centers)
    colors[i_best] = "#D62728" 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5, axis='y')
    bars = ax.bar(centers, mean_err, width=10, edgecolor="white", color=colors, yerr=std_err, lw=1, capsize=3, ecolor="k")
    
    # --- Valeur de la moyenne affichée dans chaque barre ---
    for rect, m, c in zip(bars, mean_err, counts):
        if np.isnan(m):
            continue  # bac vide
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f"{m:.3f}\n(n={c})", ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    # --------------------------------------------------------
    
    ax.set_xticks(bins)
    ax.set_xlabel('Angle between COM axis and z-axis (degrees)', fontsize=13)
    ax.set_ylabel(r'Mean absolute error on coreshell index $\left| \hat{y} - y \right|$', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=[Patch(color="#4C72B0", label="Angle interval (10°)"), Patch(color="#D62728", label="Minimum mean error")], frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig("mean_error_vs_angle.png", dpi=300, bbox_inches='tight')
    plt.close()



###############
### PROGRAM ###
###############

def main():
    data_hrtem = np.load("all_data.npy", mmap_mode="r")
    meta = np.load("meta.npy")
    if len(meta) != len(data_hrtem):
        sys.exit(f"Error: meta.npy ({len(meta)} records) and all_data.npy ({len(data_hrtem)}) do not match; run load_data.py again")

    groups = meta["id_sim"] // 10
    train_groups, test_groups = train_test_split(np.unique(groups), train_size=TRAINING_SIZE, random_state=42)
    val_groups, test_groups = train_test_split(test_groups, train_size=round(VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE), 2), random_state=42)
    test_idx = np.flatnonzero(np.isin(groups, test_groups))

    y = np.column_stack([meta[p] for p in PARAMETERS]).astype(np.float32)

    if os.path.exists(f"{DIR_NAME}/data_angle.npy"):
        angle_data = np.load(f"{DIR_NAME}/data_angle.npy") # Load precomputed angles if they exist
    else:
        id_sim = meta["id_sim"][test_idx]
        n_digits = np.floor(np.log10(id_sim)).astype(np.int64) + 1
        power = 10 ** (n_digits - 1)
        sub_database = id_sim // power # Get the sub-database number (e.g., 1 for Dataset1, 2 for Dataset2, etc.)
        id_initial = id_sim % power # Get the initial ID within the sub-database
        snapshot = id_initial // 10 + 1 # Get the snapshot number (id_sim = 0 => snapshot = 1.xyz) 
        # In HRTEM code: id_sim = (id_sim-1) * 10 + i - 1 (i=rotation index and 10 is the number of rotations per snapshot) 
    
        key = sub_database * 10**6 + snapshot # Create a unique key for each (sub_database, snapshot) pair to group them together (because the same snapshot can appear many times due to rotations)
        _, inverse = np.unique(key, return_inverse=True) # inverse give the index in the unique array for each element in key (unique, inverse = np.unique(key, reverse=True))
        order = np.argsort(inverse, kind='stable') # Give the order of indices that would sort the inverse array (kind='stable' to keep the order of equal elements)
        members_list = np.split(order, np.cumsum(np.bincount(inverse))[:-1]) # bincount counts the number of occurrences of each snapshot, cumsum gives the cumulative sum (we remove the last element) and split will separate each snapshot group
        # members_list is a list of arrays, each arrays contains the indices of the images in test_idx that belong to the same snapshot group (same sub_database and snapshot)

        n_atoms_test = meta["n_atoms"][test_idx]        
        angle = np.zeros(len(test_idx))
        phi = meta["phi"][test_idx]
        theta = meta["theta"][test_idx]
        psi = meta["psi"][test_idx]
        
        for members in members_list:
            database = sub_database[members[0]]
            snap = snapshot[members[0]]
            with open(f"/home/victor/Data/MD_Data/AgCo/Dataset{database}/XYZ/{snap}.xyz", "r") as f:
                n_atoms = int(f.readline())
                comment = f.readline()
                index1 = comment.index('Lattice="') + len('Lattice="')
                index2 = comment.index('"', index1)
                lattice = np.array(comment[index1:index2].split(), dtype=float)
                box = np.array([lattice[0], lattice[4], lattice[8]])
                pos = np.zeros((n_atoms, 3))
                rows = [f.readline().split() for _ in range(n_atoms)]
            species = np.array([r[0] for r in rows])
            pos = np.array([r[1:4] for r in rows], dtype=float)
            keep, pos = largest_cluster(pos, box)
            species = np.array(species)[keep]

            n_meta = np.unique(n_atoms_test[members])
            if len(n_meta) != 1 or len(keep) != n_meta[0]:
                sys.exit(f"Warning: Snapshot {snap} in Dataset{database} has {len(keep)} atoms in the largest cluster, but {n_meta[0]} atoms in meta.npy")
            
            mass_center_nat1 = np.mean(pos[np.where(species == 'Ag')], axis=0)
            mass_center_nat2 = np.mean(pos[np.where(species == 'Co')], axis=0)
            v = mass_center_nat2 - mass_center_nat1

            for j in members:
                cp = np.cos(phi[j])
                sp = np.sin(phi[j])
                ct = np.cos(theta[j])
                st = np.sin(theta[j])
                cs = np.cos(psi[j])
                ss = np.sin(psi[j])
                rot_matrix = np.array([
                    [cp*ct*cs - sp*ss,  -cp*ct*ss - sp*cs,  cp*st], 
                    [sp*ct*cs + cp*ss,  -sp*ct*ss + cp*cs,  sp*st], 
                    [-st*cs,            st*ss,              ct]
                ])
                v_rotated = v @ rot_matrix.T
                cos_alpha = np.abs(np.dot(v_rotated, [0, 0, 1])) / np.linalg.norm(v_rotated)
                angle[j] = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
        
        angle_data = np.zeros(len(test_idx), dtype=[('id_sim', int), ('n_atoms', int), ('composition', float), ('coreshell_index', float), ('angle', float)])
        angle_data['id_sim'] = meta["id_sim"][test_idx]
        angle_data['n_atoms'] = meta["n_atoms"][test_idx]
        angle_data['composition'] = meta["composition"][test_idx]
        angle_data['coreshell_index'] = meta["coreshell_index"][test_idx]
        angle_data['angle'] = angle
        np.save(f"{DIR_NAME}/data_angle.npy", angle_data)
        

    y_pred = pickle.load(open(f"{DIR_NAME}/predictions.pkl", "rb"))
    y_true = y[test_idx]
    err = np.abs(y_pred - y_true)
    cs_vs_angle(angle_data['angle'], y_true, err)
    error_vs_angle_hist2d(angle_data['angle'], err)
    error_vs_angle_hist2d_normalized(angle_data['angle'], err)
    plot_mean_error_vs_angle_hist(angle_data['angle'], err)



if __name__ == "__main__":
    main()
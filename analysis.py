#!/usr/bin/env python3

import pickle
import sys

import matplotlib.pyplot    as plt
import numpy                as np
import pandas               as pd

from scipy.stats                import binned_statistic_2d
from sklearn.metrics            import r2_score
from sklearn.model_selection    import train_test_split



##################
### PARAMETERS ###
##################

TRAINING_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1
PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
DIR_NAME = "resnet50_2"



############
### PLOT ###
############

def plot_training_loss(history, log_scale=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5)
    ax.plot(history["loss"], c="#4C72B0", lw=1.5, label='Training')
    ax.plot(history["val_loss"], c="#DD8452", lw=1.5, label='Validation')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
    if log_scale:
        ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    plt.tight_layout()
    plt.savefig("training_loss_log.png", dpi=300) if log_scale else plt.savefig("training_loss.png", dpi=300)
    plt.close()


def size_predictions(y_true, y_pred, corr_coeff):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5)
    ax.scatter(y_true[:,0], y_pred[:,0], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label=rf'Individual images ($R^2$ = {corr_coeff[0]:.3f})')
    ax.plot([y_true[:,0].min(), y_true[:,0].max()], [y_true[:,0].min(), y_true[:,0].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([y_true[:,0].min(), 0]), np.max([y_true[:,0].max(), 5000]))
    ax.set_ylim(np.min([y_pred[:,0].min(), 0]), np.max([y_pred[:,0].max(), 5000]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True number of atoms', fontsize=13)
    ax.set_ylabel('Predicted number of atoms', fontsize=13)
    plt.tight_layout()
    plt.savefig("predictions_n_atoms.png", dpi=300, bbox_inches='tight')
    plt.close()


def composition_predictions(y_true, y_pred, corr_coeff):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5)
    ax.scatter(y_true[:,1], y_pred[:,1], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label=rf'Individual images ($R^2$ = {corr_coeff[1]:.3f})')
    ax.plot([y_true[:,1].min(), y_true[:,1].max()], [y_true[:,1].min(), y_true[:,1].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([y_true[:,1].min(), 0]), np.max([y_true[:,1].max(), 1]))
    ax.set_ylim(np.min([y_pred[:,1].min(), 0]), np.max([y_pred[:,1].max(), 1]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True composition', fontsize=13)
    ax.set_ylabel('Predicted composition', fontsize=13)
    plt.tight_layout()
    plt.savefig("predictions_composition.png", dpi=300, bbox_inches='tight')
    plt.close()
        

def coreshell_predictions(y_true, y_pred, corr_coeff):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5)
    ax.scatter(y_true[:,2], y_pred[:,2], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label=rf'Individual images ($R^2$ = {corr_coeff[2]:.3f})')
    ax.plot([y_true[:,2].min(), y_true[:,2].max()], [y_true[:,2].min(), y_true[:,2].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([y_true[:,2].min(), -0.5]), np.max([y_true[:,2].max(), 2]))
    ax.set_ylim(np.min([y_pred[:,2].min(), -0.5]), np.max([y_pred[:,2].max(), 2]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True core-shell index', fontsize=13)
    ax.set_ylabel('Predicted core-shell index', fontsize=13)
    plt.tight_layout()
    plt.savefig("predictions_coreshell.png", dpi=300, bbox_inches='tight')
    plt.close()


def coreshell_predictions_size_sample(y_true, y_pred, xmin, xmax):

    mask = (y_true[:, 0] >= xmin) & (y_true[:, 0] <= xmax)
    y1_true = y_true[mask]
    y1_pred = y_pred[mask]
    r2 = r2_score(y1_true, y1_pred, multioutput='raw_values')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5)
    ax.plot([y_true[:,2].min(), y_true[:,2].max()], [y_true[:,2].min(), y_true[:,2].max()], c="lightgray", ls='--', lw=1.5, label='Perfect prediction')
    sc = ax.scatter(y1_true[:,2], y1_pred[:,2], c=y1_true[:,1], cmap="viridis", s=10, edgecolor='none', label=rf'Individual images ($R^2$ = {r2[2]:.3f})')
    ax.set_xlim(np.min([y_true[:,2].min(), -0.5]), np.max([y_true[:,2].max(), 2]))
    ax.set_ylim(np.min([y1_pred[:,2].min(), -0.5]), np.max([y1_pred[:,2].max(), 2]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True core-shell index', fontsize=13)
    ax.set_ylabel('Predicted core-shell index', fontsize=13)
    plt.colorbar(sc, ax=ax, label='Predicted composition')
    plt.tight_layout()
    plt.savefig(f"predictions_coreshell_{(xmax+xmin)//2}_atoms.png", dpi=300, bbox_inches='tight')
    plt.close()


def cs_vs_size(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, ls='-', lw=0.5)
    ax.scatter(y_true[:,0], y_true[:,2], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='True values')
    ax.scatter(y_pred[:,0], y_pred[:,2], c="#DD8452", s=4, alpha=.25, edgecolor='none', label='Predicted values')
    ax.set_xlim(np.min([y_true[:,0].min(), 0]), np.max([y_true[:,0].max(), 5000]))
    ax.set_ylim(np.min([y_true[:,2].min(), -0.5]), np.max([y_true[:,2].max(), 2]))
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    leg.legend_handles[1]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('Number of atoms', fontsize=13)
    ax.set_ylabel('Core-shell index', fontsize=13)
    plt.tight_layout()
    plt.savefig("coreshell_vs_size.png", dpi=300, bbox_inches='tight')
    plt.close()


def size_vs_composition_mae_normalized(y_true, err):
    stat, xedges, yedges, _ = binned_statistic_2d(y_true[:,0], y_true[:,1], np.mean(err / np.std(y_true, axis=0), axis=1), statistic='mean', bins=40)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    pcm = ax.pcolormesh(xedges, yedges, stat.T, cmap='magma', shading='auto', edgecolors='none')
    
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Normalized MAE on all parameters', fontsize=13, labelpad=10)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)
    
    ax.set_xlabel('True number of atoms', fontsize=14, labelpad=8)
    ax.set_ylabel('True composition', fontsize=14, labelpad=8)
    ax.tick_params(labelsize=11, direction='out', length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('0.3')
    ax.grid(True, color="white", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig("size_vs_composition_mae_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
def size_vs_composition_mae(y_true, err):
    
    cmap = 'magma'    
    panels = [
        (err[:, 0], 'MAE on number of atoms'),
        (err[:, 1], 'MAE on composition'),
        (err[:, 2], 'MAE on core-shell index'),
        (np.mean(err, axis=1), 'MAE on all parameters')
    ]

    fig, ax = plt.subplots(2, 2, figsize=(16, 12.5))
    for ax, (values, label) in zip(ax.flat, panels):
        stat, xedges, yedges, _ = binned_statistic_2d(y_true[:,0], y_true[:,1], values, statistic='mean', bins=40)
        pcm = ax.pcolormesh(xedges, yedges, stat.T, cmap=cmap, shading='auto')
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(label, fontsize=12, labelpad=8)
        cbar.ax.tick_params(labelsize=10)
        cbar.outline.set_linewidth(0.8)
        ax.set_xlabel('True number of atoms', fontsize=12, labelpad=6)
        ax.set_ylabel('True composition', fontsize=12, labelpad=6)
        ax.tick_params(labelsize=10, direction='out', length=4, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('0.3')
        ax.grid(True, color="white", alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)
            
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("size_vs_composition_mae.png", dpi=300, bbox_inches='tight')
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
    
    history = pd.read_csv(f"{DIR_NAME}/history.csv", index_col=0)
    y = np.column_stack([meta[p] for p in PARAMETERS]).astype(np.float32)
    y_pred = pickle.load(open(f"{DIR_NAME}/predictions.pkl", "rb"))
    y_true = y[test_idx]
    err = np.abs(y_true - y_pred)
    corr_coeff = np.diag(np.corrcoef(y_true.T, y_pred.T)[:3, 3:])
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
         
    plot_training_loss(history, log_scale=True)
    plot_training_loss(history, log_scale=False)
    size_predictions(y_true, y_pred, r2)
    composition_predictions(y_true, y_pred, r2)
    coreshell_predictions(y_true, y_pred, r2)    
    cs_vs_size(y_true, y_pred)
    
    #coreshell_predictions_size_sample(y_true, y_pred, 8, 12)
    #coreshell_predictions_size_sample(y_true, y_pred, 698, 702)
    #coreshell_predictions_size_sample(y_true, y_pred, 1398, 1402)
    #coreshell_predictions_size_sample(y_true, y_pred, 2498, 2502)
    #coreshell_predictions_size_sample(y_true, y_pred, 4998, 5002)
    #coreshell_predictions_size_sample(y_true, y_pred, 0, 100)
    #coreshell_predictions_size_sample(y_true, y_pred, 4500, 5000)
    size_vs_composition_mae_normalized(y_true, err)
    size_vs_composition_mae(y_true, err)



if __name__ == "__main__":
    main()
    
    
    
import matplotlib.pyplot as plt
import numpy as np
import os

labels = {
    "n_atoms": "number of atoms",
    "nat1": "number of Ag atoms",
    "nat2": "number of Co atoms",
    "composition": "composition (Ag fraction)",
    "coreshell_index": "coreshell index"
}

#plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    
def plot_training_loss(data, label, log_scale=False):
    """
    Plots the training and validation loss curves from the history DataFrame.
    """
    if label in labels and label in data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_axisbelow(True)
        ax.grid(alpha=.25, ls='-', lw=0.5)
        ax.plot(data[label]['train_loss'], c="#4C72B0", lw=1.5, label='Training')
        ax.plot(data[label]['val_loss'], c="#DD8452", lw=1.5, label='Validation')
        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
        if log_scale:
            ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
        
        ax.set_title(f"Training and validation loss for {labels[label]} prediction", fontsize=14, fontweight='bold', pad=12)
    else:
        exit(f"Variable '{label}' not supported for plotting training loss.")
    plt.tight_layout()
    os.makedirs(f"Figure/{label}", exist_ok=True)
    plt.savefig(f"Figure/{label}/loss_curve_{label}.png", dpi=300, bbox_inches='tight')
    plt.close()
    


def plot_predictions(data, label):
    """
    Plots the predicted vs. true values for a given variable.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=0.5)
    ax.scatter(data[label]['y_true'], data[label]['y_pred'], s=4, alpha=0.25, color="#4C72B0", edgecolor='none', label="Individual images")
    ax.plot([data[label]['y_min'], data[label]['y_max']], [data[label]['y_min'], data[label]['y_max']], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(data[label]['y_min'], data[label]['y_max'])
    ax.set_ylim(np.nanmin(data[label]['y_pred']), np.nanmax(data[label]['y_pred']))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30]  # agrandit le point dans la légende pour qu'il soit visible
    if label in labels and label in data:
        ax.set_xlabel(f"True {labels[label]}", fontsize=13)
        ax.set_ylabel(f"Predicted {labels[label]}", fontsize=13)
        ax.set_title(f"Predicted vs. true {labels[label]}", fontsize=14, fontweight='bold', pad=12)
    else:
        exit(f"Variable '{label}' not supported for plotting predictions.")
    plt.tight_layout()
    os.makedirs(f"Figure/{label}", exist_ok=True)
    plt.savefig(f"Figure/{label}/predicted_vs_true_{label}.png", dpi=300, bbox_inches='tight')
    plt.close()



""" Error histogram """

def plot_relative_error(data, label):
    n_bins = 50
    error = np.abs(data[label]['y_true'] - data[label]['y_pred'])
    relative_error = error / np.abs(data[label]['y_true']) * 100
    bins = np.linspace(data[label]['y_min'], data[label]['y_max'], n_bins + 1)
    #bins = np.linspace(100, data[label]['y_max'], n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_error = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (data[label]['y_true'] >= bins[i]) & (data[label]['y_true'] < bins[i+1])
        if np.sum(mask) > 0:
            mean_error[i] = np.mean(relative_error[mask])
        else:
            mean_error[i] = np.nan
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=0.5)
    ax.bar(bin_centers, mean_error, width=bins[1] - bins[0], align='center', alpha=0.4, color='#5B9BD5', edgecolor='#2E5C8A', lw=0.6, label="Binned mean relative error")
    ax.plot(bin_centers, mean_error, c='#1B3A5C', marker='o', markersize=3.5, lw=1.3, label="Mean relative error")
    ax.axhline(np.nanmean(mean_error), color='#E91E63', ls='--', lw=1.5, label=f"Average error : {np.nanmean(mean_error):.2f}%")
    ax.set_ylabel("Mean relative error (%)", fontsize=13)
    #ax.set_xlim(0, 5000)
    ax.set_xlim(data[label]['y_min'], data[label]['y_max'])
    #ax.set_ylim(0, min(100, np.nanmax(mean_error) * 1.3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.tick_params(labelsize=11)
    if label in labels and label in data:
        ax.set_xlabel(f"True {labels[label]}", fontsize=13)
        ax.set_title(f"Prediction error vs. true {labels[label]}", fontsize=14, fontweight='bold', pad=12)
    else:
        exit(f"Variable '{label}' not supported for plotting error histogram.")
    plt.tight_layout()
    os.makedirs(f"Figure/{label}", exist_ok=True)
    plt.savefig(f"Figure/{label}/relative_error_{label}.png", bbox_inches='tight', dpi=300)
    plt.close()
    

def plot_absolute_error(data, label):
    n_bins = 50
    error = np.abs(data[label]['y_true'] - data[label]['y_pred'])
    bins = np.linspace(data[label]['y_min'], data[label]['y_max'], n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_error = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (data[label]['y_true'] >= bins[i]) & (data[label]['y_true'] < bins[i+1])
        if np.sum(mask) > 0:
            mean_error[i] = np.mean(error[mask])
        else:
            mean_error[i] = np.nan
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=0.5)
    ax.bar(bin_centers, mean_error, width=bins[1] - bins[0], align='center', alpha=0.4, color='#5B9BD5', edgecolor='#2E5C8A', lw=0.6, label="Binned mean absolute error")
    ax.plot(bin_centers, mean_error, c='#1B3A5C', marker='o', markersize=3.5, lw=1.3, label="Mean absolute error")
    ax.axhline(np.nanmean(mean_error), color='#E91E63', ls='--', lw=1.5, label=f"Average error : {np.nanmean(mean_error):.2f} atoms")
    ax.set_ylabel("Mean absolute error", fontsize=13)
    #ax.set_xlim(0, 5000)
    ax.set_xlim(data[label]['y_min'], data[label]['y_max'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.tick_params(labelsize=11)
    if label in labels and label in data:
        ax.set_xlabel(f"True {labels[label]}", fontsize=13)
        ax.set_title(f"Prediction error vs. true {labels[label]}", fontsize=14, fontweight='bold', pad=12)
    else:
        exit(f"Variable '{label}' not supported for plotting error histogram.")
    plt.tight_layout()
    os.makedirs(f"Figure/{label}", exist_ok=True)
    plt.savefig(f"Figure/{label}/absolute_error_{label}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    

""" Distribution of the coreshell index """

    
def plot_distribution(data, label):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, linestyle='-', linewidth=0.5)
    ax.hist(data[label], bins=50, color="#4C72B0", edgecolor='#2A6F73', alpha=0.7, label=f"Dataset size: {data.shape[0]}")
    if label in labels and label in data:
        ax.set_xlabel(f"{labels[label]}", fontsize=13)
        ax.set_title(f"Distribution of the {labels[label]}", fontsize=14, fontweight='bold', pad=12)
    else:
        exit(f"Variable '{label}' not supported for plotting distribution.")
    ax.set_ylabel("Frequency", fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    plt.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    plt.tight_layout()
    os.makedirs(f"Figure/{label}", exist_ok=True)
    plt.savefig(f"Figure/{label}/distribution_{label}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
""" Comparison between predictions and computed values """

def plot_prediction_vs_computed(data, label):

    fig, ax = plt.subplots(1, 2,figsize=(14, 6.5))
    ax[0].set_axisbelow(True)
    ax[0].grid(alpha=.25, ls='-', lw=0.5)
    ax[0].scatter(data[label]['y_true'], data[label]['y_pred'], s=4, alpha=0.25, color="#4C72B0", edgecolor='none', label="Individual images")
    ax[0].plot([data[label]['y_min'], data[label]['y_max']], [data[label]['y_min'], data[label]['y_max']], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax[0].set_xlim(data[label]['y_min'], data[label]['y_max'])
    ax[0].set_ylim(np.nanmin(data[label]['y_pred']), np.nanmax(data[label]['y_pred']))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].tick_params(axis='both', labelsize=11)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    leg = ax[0].legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30]  # agrandit le point dans la légende pour qu'il soit visible
    if label in labels and label in data:
        ax[0].set_xlabel(f"True {labels[label]}", fontsize=13)
        ax[0].set_ylabel(f"Predicted {labels[label]}", fontsize=13)
        ax[0].set_title("Predicted", fontsize=14, fontweight='bold', pad=12)
    else:
        exit(f"Variable '{label}' not supported for plotting predictions.")

    ax[1].set_axisbelow(True)
    ax[1].grid(alpha=.25, ls='-', lw=0.5)
    ax[1].scatter(data[label]['y_true'], data[label]['y_computed'], s=4, alpha=0.25, color="#4C72B0", edgecolor='none', label="Individual images")
    ax[1].plot([data[label]['y_min'], data[label]['y_max']], [data[label]['y_min'], data[label]['y_max']], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax[1].set_xlim(data[label]['y_min'], data[label]['y_max'])
    ax[1].set_ylim(np.nanmin(data[label]['y_computed']), np.nanmax(data[label]['y_computed']))
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].tick_params(axis='both', labelsize=11)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    leg = ax[1].legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30]  # agrandit le point dans la légende pour qu'il soit visible
    if label in labels and label in data:
        ax[1].set_xlabel(f"True {labels[label]}", fontsize=13)
        ax[1].set_ylabel(f"Predicted {labels[label]}", fontsize=13)
        ax[1].set_title("Computed", fontsize=14, fontweight='bold', pad=12)
        fig.suptitle(f"Direct prediction vs. computed value - {labels[label].capitalize()}", fontsize=15, fontweight='bold', y=1.02)
    else:
        exit(f"Variable '{label}' not supported for plotting predictions.")
    plt.tight_layout()
    os.makedirs(f"Figure/{label}", exist_ok=True)
    plt.savefig(f"Figure/{label}/predicted_vs_computed_{label}.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_coreshell_index_vs_composition(data, pred=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=0.5)
    if pred:
        ax.scatter(data['composition']['y_pred'], data['coreshell_index']['y_pred'], s=4, alpha=0.25, c="#4C72B0", edgecolor='none', label="Individual images")
    else:
        ax.scatter(data['composition']['y_true'], data['coreshell_index']['y_true'], s=4, alpha=0.25, c="#4C72B0", edgecolor='none', label="Individual images")
    ax.set_xlabel('Composition', fontsize=13)
    ax.set_ylabel('Coreshell Index', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.set_title("Coreshell Index as a Function of Composition", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    os.makedirs("Figure", exist_ok=True)
    if pred:
        plt.savefig(f"Figure/coreshell_index_vs_composition_pred.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"Figure/coreshell_index_vs_composition_true.png", dpi=300, bbox_inches='tight')
    plt.close()


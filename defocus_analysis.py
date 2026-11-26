#!/usr/bin/env python3

import os
import pickle
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#from scipy.stats import binned_statistic_2d
from matplotlib.lines import Line2D


##################
### PARAMETERS ###
##################

TRAINING_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1
BATCH_SIZE = 128
PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
FIELDS = ["id_sim"] + PARAMETERS
CHUNK = 100000
DIR_NAME = "all_data3"


tf.keras.mixed_precision.set_global_policy("mixed_float16") # float16 on the V100 tensor cores, weights kept in float32



###############
### CLASSES ###
###############

class Batches(tf.keras.utils.PyDataset):
    # Reads one batch at a time from the memory-mapped file; images stay uint8 until the Rescaling layer, on the GPU
    def __init__(self, images, y, idx, shuffle):
        super().__init__(workers=4)
        self.images, self.y, self.idx, self.shuffle = images, y, idx.copy(), shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / BATCH_SIZE))

    def __getitem__(self, i):
        idx = np.sort(self.idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        return self.images[idx][..., np.newaxis], self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


@tf.keras.utils.register_keras_serializable(package="Neural_network")
class RandomTranspose(tf.keras.layers.Layer):
    # Swaps x and y in half of the images, during training only; with RandomFlip this gives the 8 symmetries of the square
    def call(self, x, training=None):
        if not training:
            return x
        swap = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.5
        return tf.where(swap, tf.transpose(x, [0, 2, 1, 3]), x)


@tf.keras.utils.register_keras_serializable(package="Neural_network")
class Denormalize(tf.keras.layers.Layer):
    # Min-max normalised outputs back to physical units; the parameters flagged in cube were fitted as their cube root
    def __init__(self, y_min, y_max, cube, **kwargs):
        super().__init__(**kwargs)
        self.y_min, self.y_max, self.cube = [float(v) for v in y_min], [float(v) for v in y_max], [bool(c) for c in cube]

    def call(self, y):
        y = y * (tf.constant(self.y_max) - tf.constant(self.y_min)) + tf.constant(self.y_min)
        return tf.where(tf.constant(self.cube), y ** 3, y)

    def get_config(self):
        return {**super().get_config(), "y_min": self.y_min, "y_max": self.y_max, "cube": self.cube}



############
### PLOT ###
############

def error_vs_defocus_c3(defocus, c3, err):
    c3_vals = np.unique(c3) * 1e-3
    df_vals = np.unique(defocus)
    ci = np.searchsorted(c3_vals, c3 * 1e-3)
    di = np.searchsorted(df_vals, defocus)
    grid = np.full((len(c3_vals), len(df_vals)), np.nan)
    grid[ci, di] = err[:, 2]

    # Meilleur defocus pour chaque C3 (minimum de l'erreur le long de l'axe defocus)
    valid = ~np.all(np.isnan(grid), axis=1)
    j_min = np.nanargmin(grid[valid], axis=1)
    c3_best = c3_vals[valid]
    df_best = df_vals[j_min]

    ht = 200.0
    hc = 1.2398419843320026
    e0 = 510.99895069
    wavelength = hc / np.sqrt(ht * (2.0 * e0 + ht))
    df_scherzer = -1.2 * np.sqrt(abs(c3_vals * 1e3 * wavelength))
    df_scherzer_best = -1.2 * np.sqrt(abs(c3_best * 1e3 * wavelength))
    tol = np.median(np.diff(df_vals))
    near = np.abs(df_best - df_scherzer_best) <= tol

    dx = np.median(np.diff(c3_vals))
    dy = np.median(np.diff(df_vals))
    extent = [c3_vals[0] - dx / 2, c3_vals[-1] + dx / 2, df_vals[0] - dy / 2, df_vals[-1] + dy / 2]

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    mesh = ax.imshow(grid.T, cmap="viridis", aspect="auto", origin="lower", extent=extent, interpolation="nearest")
    ax.plot(c3_vals, df_scherzer, color="black", lw=1.8, ls=(0, (6, 3)), zorder=4)
    ax.scatter(c3_best[~near], df_best[~near], s=28, marker="o", facecolors="white", edgecolor="black", lw=0.6, zorder=5, alpha=0.85)
    ax.scatter(c3_best[near], df_best[near], s=70, marker="D", facecolors="#ff2d95", edgecolor="black", lw=0.8, zorder=6)
    ax.grid(alpha=0.3, c="k", zorder=5)
    ax.set_xlabel("Spherical aberration (µm)")
    ax.set_ylabel("Defocus (nm)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=0.6, markersize=6, label="Best defocus per C3"),
        Line2D([], [], marker="D", ls="", mfc="#ff2d95", mec="black", mew=0.8, markersize=8, label=rf"Best defocus within $\pm${tol:g} nm of Scherzer"),
        Line2D([], [], color="black", lw=1.8, ls=(0, (6, 3)), label="Scherzer defocus")
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, edgecolor="lightgray")
    fig.colorbar(mesh, ax=ax, pad=0.02, label=r"Error on core-shell index prediction $\left| \hat{y} - y \right|$")
    plt.savefig("error_vs_defocus_c3.png", dpi=300)
    plt.close()



###############
### PROGRAM ###
###############

def main():
    data_hrtem = np.load("all_data.npy", mmap_mode="r")
    #meta = np.load("meta.npy")
    meta = np.empty(len(data_hrtem), dtype=[(f, data_hrtem.dtype[f]) for f in FIELDS])
    for start in range(0, len(data_hrtem), CHUNK):
        chunk = data_hrtem[start:start + CHUNK]
        for f in FIELDS:
            meta[f][start:start + CHUNK] = chunk[f]    

    if len(meta) != len(data_hrtem):
        sys.exit(f"Error: meta.npy ({len(meta)} records) and all_data.npy ({len(data_hrtem)}) do not match; run load_data.py again")

    groups = meta["id_sim"] // 10
    train_groups, test_groups = train_test_split(np.unique(groups), train_size=TRAINING_SIZE, random_state=42)
    val_groups, test_groups = train_test_split(test_groups, train_size=round(VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE), 2), random_state=42)
    train_idx, val_idx, test_idx = (np.flatnonzero(np.isin(groups, g)) for g in (train_groups, val_groups, test_groups))

    y = np.column_stack([meta[p] for p in PARAMETERS]).astype(np.float32)
    y_fit = np.column_stack([np.cbrt(y[:, i]) if p == "n_atoms" else y[:, i] for i, p in enumerate(PARAMETERS)]) # N^(1/3) scales like the particle size in the image
    y_min = np.min(y_fit[train_idx], axis=0)
    y_max = np.max(y_fit[train_idx], axis=0)

    model = tf.keras.models.load_model(f"{DIR_NAME}/model.keras")
    
    #meta_angle = np.load("meta_angle.npy")
    #data_angle = np.load("data_angle.npy")
    data_defocus = np.load("data_defocus.npy")
    y_defocus = np.column_stack([data_defocus[p] for p in PARAMETERS]).astype(np.float32)
    y_fit_defocus = np.column_stack([np.cbrt(y_defocus[:, i]) if p == "n_atoms" else y_defocus[:, i] for i, p in enumerate(PARAMETERS)]) # N^(1/3) scales like the particle size in the image
    y_norm = (y_fit_defocus - y_min) / (y_max - y_min)

    images_defocus = data_defocus["image"]
    defocus_gen = Batches(images_defocus, y_norm, np.arange(len(images_defocus)), shuffle=False)

    if os.path.exists(f"{DIR_NAME}/predictions_defocus.npy"):
        predictions_defocus = np.load(f"{DIR_NAME}/predictions_defocus.npy")
    else:
        predictions_defocus = model.predict(defocus_gen, verbose=1)
        np.save(f"{DIR_NAME}/predictions_defocus.npy", predictions_defocus)

    y_true = np.array([y_defocus[:, i] for i, p in enumerate(PARAMETERS)]).T
    y_pred =  predictions_defocus
    err = np.abs(y_pred - y_true)
    defocus = np.array(data_defocus["defocus"])
    spherical_aberration = np.array(data_defocus["C3"])
    #alpha = np.array(meta_angle["angle"])
    
    error_vs_defocus_c3(defocus, spherical_aberration, err)
    
    
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


labels = ["Number of atoms", "Composition", "Core-shell index"]
r2_allara = np.array([0.957, 0.993, 0.942]) # Allara et al. 2026
#r2_archi1 = np.array([0.990, 0.954, 0.834]) # Architecture 1
#r2_archi2 = np.array([0.984, 0.935, 0.798]) # Architecture 2
r2_resnet50_30 = np.array([0.991, 0.958, 0.841]) # ResNet50 (30 epochs)
r2_resnet50_60 = np.array([0.992, 0.961, 0.848]) # ResNet50 (60 epochs)

series = [
    (r2_allara, "#4C72B0", "X-Ray Scattering (Allara et al. 2026)"),
    #(r2_archi1, "#DD8452", "HRTEM with Architecture 1"),
    #(r2_archi2, "#55A868", "HRTEM with Architecture 2"),
    (r2_resnet50_30, "#C44E52", "HRTEM with ResNet50 (30 epochs)"),
    (r2_resnet50_60, "#DD8452", "HRTEM with ResNet50 (60 epochs)")
]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(6, 4.2))
for i in x[::2]:
    ax.axvspan(i - 0.5, i + 0.5, color="0.5", alpha=0.06, zorder=0, lw=0)

for r2, color, label in series:
    ax.scatter(x, r2, color=color, alpha=0.85, s=70, label=label, zorder=3, edgecolors="white", lw=0.8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylabel(r"$R^2$", fontsize=13)
ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("0.4")
ax.tick_params(axis="both", colors="0.2", length=3)
ax.legend(fontsize=9.5, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, handletextpad=0.3, columnspacing=1.5)
plt.savefig("r2_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
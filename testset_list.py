#!/usr/bin/env python3

import os
import shutil
import time

import numpy as np

from sklearn.model_selection import train_test_split



##################
### PARAMETERS ###
##################

TRAINING_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1
PATH = "/home/victor/Data/MD_Data/AgCo"
#OUT = "./testset"
OUT = "/home/victor/Data/MD_Data/testset"



###############
### PROGRAM ###
###############

def main():
    start_time = time.time()
    meta = np.load("meta.npy")
    groups = meta["id_sim"] // 10
    train_groups, test_groups = train_test_split(np.unique(groups), train_size=TRAINING_SIZE, random_state=42)
    val_groups, test_groups = train_test_split(test_groups, train_size=round(VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE), 2), random_state=42)
    test_idx = np.flatnonzero(np.isin(groups, test_groups))

    m = meta[test_idx]
    id_sim = m["id_sim"]
    n_digits = np.floor(np.log10(id_sim)).astype(np.int64) + 1
    power = 10 ** (n_digits - 1)
    sub_database = id_sim // power
    snapshot = (id_sim % power) // 10 + 1

    keys = np.column_stack([sub_database, snapshot])
    pairs, first = np.unique(keys, axis=0, return_index=True)
    lines = {}
    for (sub, snap), i in zip(pairs, first):
        dst = f"{OUT}/Dataset{sub}/XYZ"
        src = os.path.join(PATH, f"Dataset{sub}", "XYZ", f"{snap}.xyz")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(src, dst)
        with open(src) as f:
            n_atoms_total = int(f.readline())
        r = m[i]
        lines.setdefault(sub, []).append(f"{snap} {r['n_atoms']} {r['composition']} {r['n_steps']} {r['initial_temperature']}")

    for sub, l in lines.items():
        with open(f"{OUT}/Dataset{sub}/data.dat", "w") as f:
            f.write("\n".join(l))
            
    print(f"{len(pairs)} fichiers copiés")
    print(f"Temps d'exécution : {time.time() - start_time:.2f} secondes")

if __name__ == "__main__":
    main()
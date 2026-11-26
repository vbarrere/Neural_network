#!/usr/bin/env python3

import time
import numpy as np

PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
FIELDS = ["id_sim"] + PARAMETERS
CHUNK = 100000

start_time = time.time()
data = np.load("all_data.npy", mmap_mode="r")
meta = np.empty(len(data), dtype=[(f, data.dtype[f]) for f in FIELDS])
for start in range(0, len(data), CHUNK):
    chunk = data[start:start + CHUNK]
    for f in FIELDS:
        meta[f][start:start + CHUNK] = chunk[f]

np.save("meta.npy", meta)
print("Time to save data: ", time.time() - start_time, flush=True)
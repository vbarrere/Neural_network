#!/usr/bin/bash

export hrtem_path="/home/victor/Data/HRTEM_data/AgCo/"
export save_data_path="/home/victor/Data/Neural_network_data/"


export epochs=200
export patience=20
export n_samples=400000 # Set to -1 to use all samples, or specify a number for a subset


./main.py
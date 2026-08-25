#!/usr/bin/bash

export hrtem_path="/home/victor/Data/HRTEM_data_test/AgCo/"
export save_data_path="/home/victor/Data/Predictions_data/"


#export epochs=200
export epochs=50

#export patience=20
export patience=10
export n_samples=500000 # Set to -1 to use all samples, or specify a number for a subset
export n_px=128

./main.py
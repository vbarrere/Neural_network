#!/usr/bin/env python3

import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt


data = pd.read_csv("/home/victor.barrere@crmd.cnrs-orleans.fr/Documents/Data/Data_processed/data.dat", sep="\t", engine="python", na_values=["nan"], header=None)
data.columns = ["i_sim", "n_atoms", "nat1", "nat2", "n_steps", "initial_temperature", "epot", "surface_area", "solid_volume", "cna_others", "cna_fcc", "cna_hcp", "cna_bcc", "cna_ico", "bond_angle_others", "bond_angle_fcc", "bond_angle_hcp", "bond_angle_bcc", "bond_angle_ico", "d_com", "gyration_radius", "n_atm1_out", "n_atm2_out", "n_atm1_in", "n_atm2_in", "r_cm_x", "r_cm_y", "r_cm_z", "r_cm1_x", "r_cm1_y", "r_cm1_z", "r_cm2_x", "r_cm2_y", "r_cm2_z", "csp"]

data["image_file"] =  data["i_sim"] + '.png'
i_sim = data["i_sim"]
n_atoms = data["n_atoms"].values
nat1 = data["nat1"].values
nat2 = data["nat2"].values
n_steps = data["n_steps"].values
initial_temperature = data["initial_temperature"].values
epot = data["epot"].values
surface_area = data["surface_area"].values
solid_volume = data["solid_volume"].values
cna_others = data["cna_others"].values
cna_fcc = data["cna_fcc"].values
cna_hcp = data["cna_hcp"].values
cna_bcc = data["cna_bcc"].values
cna_ico = data["cna_ico"].values
bond_angle_others = data["bond_angle_others"].values
bond_angle_fcc = data["bond_angle_fcc"].values
bond_angle_hcp = data["bond_angle_hcp"].values
bond_angle_bcc = data["bond_angle_bcc"].values
bond_angle_ico = data["bond_angle_ico"].values
d_com = data["d_com"].values
gyration_radius = data["gyration_radius"].values
n_atm1_out = data["n_atm1_out"].values
n_atm2_out = data["n_atm2_out"].values
n_atm1_in = data["n_atm1_in"].values
n_atm2_in = data["n_atm2_in"].values
r_cm_x = data["r_cm_x"].values
r_cm_y = data["r_cm_y"].values
r_cm_z = data["r_cm_z"].values
r_cm1_x = data["r_cm1_x"].values
r_cm1_y = data["r_cm1_y"].values
r_cm1_z = data["r_cm1_z"].values
r_cm2_x = data["r_cm2_x"].values
r_cm2_y = data["r_cm2_y"].values
r_cm2_z = data["r_cm2_z"].values
csp = data["csp"].values

r_cm = np.array([r_cm_x, r_cm_y, r_cm_z]).T
r_cm1 = np.array([r_cm1_x, r_cm1_y, r_cm1_z]).T
r_cm2 = np.array([r_cm2_x, r_cm2_y, r_cm2_z]).T


eta_parameter = 2 * np.abs(n_atm1_out / (n_atm1_out + n_atm2_out) - nat1 / n_atoms) + 2 * np.abs(n_atm1_in / (n_atm1_in + n_atm2_in) - nat1 / n_atoms) - d_com / (2*gyration_radius)

img_path = "/home/victor.barrere@crmd.cnrs-orleans.fr/Documents/Data/Data_HRTEM/HRTEM_image"
img_list = glob.glob(img_path + "/*.png")

images = np.zeros((len(img_list), 64, 64, 1))
id = []
for i in range(len(img_list)):
    img = PIL.Image.open(img_list[i]).convert("L").resize((64, 64))
    images[i] = np.array(img).reshape(64, 64, 1)
    id.append(img_list[i].split("/")[-1].split(".")[0])
id = np.array(id)


index = np.argsort(i_sim)
i_sim = np.sort(i_sim)
eta_parameter = eta_parameter[index]

index = np.argsort(id)
id = np.sort(id)
images = images[index]

#data = data.iloc[index]


x_train = images[:int(0.8*len(id)), :, :, :]
y_train = eta_parameter[:int(0.8*len(id))]

x_val = images[int(0.8*len(id)):int(0.9*len(id)), :, :, :]
y_val = eta_parameter[int(0.8*len(id)):int(0.9*len(id))]

x_test =images[int(0.9*len(id)):, :, :, :]
y_test = eta_parameter[int(0.9*len(id)):]


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(64, 64, 1)))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(50, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="linear"))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


predictions = model.predict(x_test)
predictions = predictions.flatten()

"""
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
"""

plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title("True vs Predicted Values")
plt.show()
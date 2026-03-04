#!/usr/bin/env python3

import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

img_path = "/home/victor.barrere@crmd.cnrs-orleans.fr/Documents/Data/Data_HRTEM/HRTEM_image"
data = pd.read_csv("/home/victor.barrere@crmd.cnrs-orleans.fr/Documents/Data/Data_processed/data.dat", sep="\t", engine="python", na_values=["nan"], header=None)

data.columns = ["i_sim", "n_atoms", "nat1", "nat2", "n_steps", "initial_temperature", "epot", "surface_area", "solid_volume", "cna_others", "cna_fcc", "cna_hcp", "cna_bcc", "cna_ico", "bond_angle_others", "bond_angle_fcc", "bond_angle_hcp", "bond_angle_bcc", "bond_angle_ico", "d_com", "gyration_radius", "nat1_out", "nat2_out", "nat1_in", "nat2_in", "r_cm_x", "r_cm_y", "r_cm_z", "r_cm1_x", "r_cm1_y", "r_cm1_z", "r_cm2_x", "r_cm2_y", "r_cm2_z", "csp"]
data["image_file"] =  data["i_sim"] + '.png'
data["eta_parameter"] = 2 * np.abs(data["nat1_out"] / (data["nat1_out"] + data["nat2_out"]) - data["nat1"] / data["n_atoms"]) + 2 * np.abs(data["nat1_in"] / (data["nat1_in"] + data["nat2_in"]) - data["nat1"] / data["n_atoms"]) - data["d_com"] / (2*data["gyration_radius"])

mask = np.isnan(data["eta_parameter"])

train_data, test_data = train_test_split(data[~mask], test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_data, 
    directory=img_path, 
    x_col="image_file", 
    y_col="eta_parameter",
    class_mode='raw',
    target_size=(64, 64),
    color_mode="grayscale"
)

val_gen = datagen.flow_from_dataframe (
    dataframe=val_data, 
    directory=img_path, 
    x_col="image_file", 
    y_col="eta_parameter",
    class_mode='raw',
    target_size=(64, 64),
    color_mode="grayscale"
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe (
    dataframe=test_data, 
    directory=img_path, 
    x_col="image_file", 
    y_col="eta_parameter",
    class_mode='raw',
    target_size=(64, 64),
    color_mode="grayscale"
)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(64, 64, 1)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(768, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_gen, epochs=10, validation_data=val_gen)


predictions = model.predict(test_gen)
predictions = predictions.flatten()

y_true = test_gen.labels

plt.scatter(y_true, predictions, alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title("True vs Predicted Values")
plt.show()
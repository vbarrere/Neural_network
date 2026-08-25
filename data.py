import os

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from PIL import Image

def image_line_to_png(image_file, out_dir, index, nx=96):
    os.makedirs(out_dir, exist_ok=True)
    with open(image_file) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            id_sim, _, pixels = line.split(" ", 2)
            img = np.fromstring(pixels, sep=" ", dtype=np.int16).astype(np.int16)
            img = (img + 128).astype(np.uint8).reshape(nx, nx)
            out_png = os.path.join(out_dir, f"{id_sim}_{index}.png")
            Image.fromarray(img, mode="L").save(out_png)
            print(f"wrote {out_png}")  
        

def load_nanoalloys_data(path, size_sample=-1):
    try:
        df_tmp = {}
        n_dataset = 4
        for i in range(n_dataset):
            df_tmp[i] = pd.read_csv(path + f"Dataset{i+1}/data.dat", sep=r'\s+', na_values=["nan"], header=0)
            df_tmp[i].columns = ['id_sim', 'n_atoms', 'n_steps', 'initial_temperature', 'epot_total', 'composition', 'gyration_radius', 'nat1', 'nat2', 'nat1_out', 'nat2_out', 'nat1_in', 'nat2_in', 'd_com', 'coreshell_index']
            df_tmp[i]['id_sim'] += f"_{i+1}"
        df = pd.concat([df_tmp[0], df_tmp[1], df_tmp[2], df_tmp[3]], ignore_index=True)
        df.columns = ['id_sim', 'n_atoms', 'n_steps', 'initial_temperature', 'epot_total', 'composition', 'gyration_radius', 'nat1', 'nat2', 'nat1_out', 'nat2_out', 'nat1_in', 'nat2_in', 'd_com', 'coreshell_index']
        df["image_file"] = df["id_sim"] + ".png"
        df["coreshell_index"] = 2 * np.abs(df["nat1_out"] / (df["nat1_out"] + df["nat2_out"]) - df["nat1"] / df["n_atoms"]) + 2 * np.abs(df["nat1_in"] / (df["nat1_in"] + df["nat2_in"]) - df["nat1"] / df["n_atoms"]) - df["d_com"] / (2*df["gyration_radius"])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df[~np.isnan(df["coreshell_index"])]
        print(f"Loaded data with {len(df)} samples from {path}")
        if size_sample != -1:
            df = df.head(size_sample)
        return df
    except FileNotFoundError:
        exit(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        exit(f"The data file is empty: {path}")
    except Exception as e:
        exit(f"An error occurred while reading the data file: {e}")



def split_data(df, data, var, train_size=0.6, val_size=0.3, test_size=0.1, random_state=42):
    training_set, test_set = train_test_split(df, train_size=train_size, random_state=random_state)
    val_size_adjusted = round(val_size / (val_size + test_size), 2)
    validation_set, test_set = train_test_split(test_set, train_size=val_size_adjusted, random_state=random_state)
    if var in df.columns:
        y_train = training_set[var].values
        data[var]['y_min'], data[var]['y_max'] = np.min(y_train), np.max(y_train)

        training_set[var] = (training_set[var] - data[var]['y_min']) / (data[var]['y_max'] - data[var]['y_min'])
        validation_set[var] = (validation_set[var] - data[var]['y_min']) / (data[var]['y_max'] - data[var]['y_min'])
        test_set[var] = (test_set[var] - data[var]['y_min']) / (data[var]['y_max'] - data[var]['y_min'])
    else:
        exit(f"Variable '{var}' not found in the data frame columns.")
    return training_set, validation_set, test_set



def create_generator(training_set, validation_set, test_set, path, output, n_px):
    
    datagen_training = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    training_gen = datagen_training.flow_from_dataframe(
        dataframe=training_set,
        directory=path,
        x_col="image_file",
        y_col=[output],
        target_size=(n_px, n_px),
        batch_size=256,
        class_mode='raw',
        color_mode="grayscale"
    )
    validation_gen = datagen.flow_from_dataframe(
        dataframe=validation_set,
        directory=path,
        x_col="image_file",
        y_col=[output],
        target_size=(n_px, n_px),
        batch_size=256,
        class_mode='raw',
        color_mode="grayscale"
    )
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_set,
        directory=path,
        x_col="image_file",
        y_col=[output],
        target_size=(n_px, n_px),
        batch_size=256,
        class_mode='raw',
        color_mode="grayscale",
        shuffle=False
    )
    return training_gen, validation_gen, test_gen


def compute_data(data):
    if 'n_atoms' not in data or 'nat1' not in data:
        exit("Input DataFrame must contain 'n_atoms' and 'nat1' columns.")
    data['nat2']['y_computed'] = data['n_atoms']['y_pred'] - data['nat1']['y_pred']
    data['composition']['y_computed'] = data['nat1']['y_pred'] / data['n_atoms']['y_pred']



def load_predictions(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def save_predictions(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
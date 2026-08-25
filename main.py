#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Masque les messages I et W
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from neural_network import convolutional_neural_network, network, convolutional_neural_network2
from plot import plot_training_loss, plot_predictions, plot_relative_error, plot_absolute_error, plot_distribution, plot_prediction_vs_computed, plot_coreshell_index_vs_composition
from data import load_nanoalloys_data, split_data, create_generator, load_predictions, save_predictions, compute_data
import numpy as np

hrtem_path = os.getenv("hrtem_path")
save_data_path = os.getenv("save_data_path")
n_epochs = int(os.getenv("epochs"))
patience = int(os.getenv("patience"))
n_samples = int(os.getenv("n_samples"))
n_px = int(os.getenv("n_px"))
param = ["nat1", "n_atoms"]

df = load_nanoalloys_data(hrtem_path, size_sample=n_samples)
prediction_file = save_data_path + f"{n_epochs}_epochs_{n_samples}/predictions.pkl"
if os.path.exists(prediction_file):
    data = load_predictions(prediction_file)
else:
    data = {}

for var in param:
    print("--------------------------------------------------")        
    print(f"Processing variable: {var}\n")

    if var not in data:
        data[var] = {}
        train_set, val_set, test_set = split_data(df, data, var, train_size=0.6, val_size=0.3, test_size=0.1, random_state=42)
        train_gen, val_gen, test_gen = create_generator(train_set, val_set, test_set, hrtem_path + "images_png/", var, n_px)
        modele = convolutional_neural_network(n_px)
        history = network(train_gen, val_gen, modele, n_epochs, patience=patience)

        data[var]['y_true'] = test_set[var].values * (data[var]['y_max'] - data[var]['y_min']) + data[var]['y_min']
        data[var]['y_pred'] = modele.predict(test_gen)      
        data[var]['y_pred'] = data[var]['y_pred'].flatten()
        data[var]['y_pred'] = data[var]['y_pred'] * (data[var]['y_max'] - data[var]['y_min']) + data[var]['y_min']
        data[var]['train_loss'] = history.history['loss']
        data[var]['val_loss'] = history.history['val_loss']
        data[var]['train_mse'] = modele.evaluate(train_gen)
        data[var]['val_mse'] = modele.evaluate(val_gen)
        data[var]['test_mse'] = modele.evaluate(test_gen)
        data[var]['id_sim'] = test_set['id_sim'].values

    print(f"Train MSE: {data[var]['train_mse']}")
    print(f"Validation MSE: {data[var]['val_mse']}")
    print(f"Test MSE: {data[var]['test_mse']}")

    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    save_predictions(prediction_file, data)  

    plot_training_loss(data, var, log_scale=False)
    plot_predictions(data, var)
    plot_relative_error(data, var)
    plot_absolute_error(data, var)

    print("--------------------------------------------------\n")    


plot_distribution(df, "coreshell_index")
#compute_data(data)

#plot_prediction_vs_computed(data, "nat2")
#plot_prediction_vs_computed(data, "composition")
#plot_coreshell_index_vs_composition(data, pred=True)
#plot_coreshell_index_vs_composition(data, pred=False)


def print_baseline_comparison(data, var):
    """
    Compare le modèle à une baseline naïve (prédire la moyenne),
    en s'assurant que les deux MSE sont calculées sur la même échelle
    (valeurs dénormalisées), contrairement à modele.evaluate() qui travaille
    sur les valeurs normalisées 0-1.
    """
    y_true = data[var]['y_true']
    y_pred = data[var]['y_pred']

    # MSE réelle du modèle, en unités dénormalisées (comparable à la baseline)
    model_mse = np.mean((y_true - y_pred) ** 2)

    # Baseline : prédire la moyenne (ici approximée par la moyenne du test set)
    baseline_mse = np.var(y_true)

    print(f"[{var}]")
    print(f"Baseline MSE (prédire la moyenne) : {baseline_mse:.5f}")
    print(f"MSE du modèle (dénormalisée)      : {model_mse:.5f}")
    print(f"Ratio (plus petit = mieux)        : {model_mse / baseline_mse:.3f}")
    print()


print("\n--------------------------------------------------\n")
print_baseline_comparison(data, 'composition')
print_baseline_comparison(data, 'n_atoms')
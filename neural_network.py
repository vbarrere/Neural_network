import tensorflow as tf
from tensorflow.keras import regularizers

def network(training_data, validation_data, modele, epochs, patience):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6)
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    modele.compile(loss='mean_squared_error', optimizer=optimizer)
    history = modele.fit(training_data, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
    return history


def convolutional_neural_network(dropout_rate=.3, l2_reg=1e-4):
    """
    Creates a convolutional neural network for image regression.
    """
    modele = tf.keras.models.Sequential()
    modele.add(tf.keras.layers.Input(shape=(96, 96, 1)))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    modele.add(tf.keras.layers.Dropout(dropout_rate))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    modele.add(tf.keras.layers.Dropout(dropout_rate))
    modele.add(tf.keras.layers.Flatten())
    modele.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    modele.add(tf.keras.layers.Dropout(dropout_rate))
    modele.add(tf.keras.layers.Dense(768, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    modele.add(tf.keras.layers.Dropout(dropout_rate))
    modele.add(tf.keras.layers.Dense(1, activation='linear'))
    
    return modele

def convolutional_neural_network2():
    """
    Creates a convolutional neural network for image regression.
    """
    modele = tf.keras.models.Sequential()
    modele.add(tf.keras.layers.Input(shape=(96, 96, 1)))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    modele.add(tf.keras.layers.Flatten())
    modele.add(tf.keras.layers.Dense(1024, activation='relu'))
    modele.add(tf.keras.layers.Dense(768, activation='relu'))
    modele.add(tf.keras.layers.Dense(1, activation='linear'))
    
    return modele


def u_net():
    """
    Creates a U-Net model for image regression.
    """
    pass  # Placeholder for U-Net implementation
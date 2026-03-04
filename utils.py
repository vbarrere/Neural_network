import tensorflow as tf

def init_gen(train_data, val_data, test_data, img_path, x, y):
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_data, 
        directory=img_path, 
        x_col=x, 
        y_col=y,
        class_mode='raw',
        target_size=(64, 64),
        color_mode="grayscale"
    )

    val_gen = datagen.flow_from_dataframe (
        dataframe=val_data, 
        directory=img_path, 
        x_col=x, 
        y_col=y,
        class_mode='raw',
        target_size=(64, 64),
        color_mode="grayscale"
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_dataframe (
        dataframe=test_data, 
        directory=img_path, 
        x_col=x, 
        y_col=y,
        class_mode='raw',
        target_size=(64, 64),
        color_mode="grayscale"
    )
    
    return train_gen, val_gen, test_gen
    
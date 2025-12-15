
from sklearn.metrics import confusion_matrix

# import tensorflow
import tensorflow as tf
import numpy as np

# TensorFLow Tools
import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint
# from livelossplot.tf_keras import PlotLossesCallback


def prototype_model():
    inputs = tf.keras.layers.Input((150, 1))

    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    outputs = tf.keras.layers.Dense(5, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    losses = tf.keras.losses.SparseCategoricalCrossentropy(),
    model.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
    return model


def model_version_2(input_length=150, num_classes=5):


    inputs = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
    model.summary()
    return model


def model_version_3(signal_length=150, num_classes=5):
    inputs = layers.Input(shape=(signal_length, 1))

    x = layers.Conv1D(32, kernel_size=25, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=15, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(pool_size=2)(x)   # 150 → 75
    x = layers.Dropout(0.25)(x)


    x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(pool_size=2)(x)   # 75 → 37
    x = layers.Dropout(0.30)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)


    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Dense(64, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model



def model_version_4(signal_length=(100,), num_classes=5):
    inputs = layers.Input(shape=signal_length)

    x = layers.Normalization(axis=None)(inputs)  # normalise input
    x = layers.MaxPooling1D(pool_size=4)(x)  

    x = layers.Conv1D(20, kernel_size=31, padding='same', activation='relu')(x)  # large kernel for big features
    x = layers.Conv1D(50, kernel_size=15, padding='same', activation='relu')(x)  # medium kernel for medium sized features
    x = layers.Conv1D(150, kernel_size=3, padding='same', activation='relu')(x)  # small kernel for details. 

    x = layers.Flatten()(x)

    x = layers.Dense(80, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model


def model_version_5(signal_length=(100,), num_classes=5):
    inputs = layers.Input(shape=signal_length)

    # x = layers.Normalization(axis=1)(inputs)  # normalise input
    # x = layers.MaxPooling1D(pool_size=4)(inputs)  
    
    x = inputs

    # x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)

    conv1 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)   # small kernel for details.
    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)  # medium kernel for medium sized features
    conv3 = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(x)  # large kernel for big features

    x = layers.Concatenate()([conv1, conv2, conv3])
    x = layers.Conv1D(64, 5, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model
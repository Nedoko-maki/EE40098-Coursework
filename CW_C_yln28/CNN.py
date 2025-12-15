
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


def build_peaknet_1d(input_length=150, num_classes=5):

    inp = layers.Input(shape=(input_length, 1))

    # First convolutional block
    x = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp)
    x = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Second convolutional block
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Third convolutional block
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out)

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
    model.summary()
    return model
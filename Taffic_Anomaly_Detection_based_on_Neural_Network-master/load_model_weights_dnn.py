import tensorflow as tf
from tensorflow.keras import utils,layers
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json


def create_Dnn(h5_path):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)
    model.load_weights(h5_path,
                       by_name=True
                       )
    return model
# model=create_Dnn("dnn_model.h5")
model=load_model("dnn_model")




from typing import Optional
from empkins_io.datasets.d03.micro_gapvii._dataset import MicroBaseDataset

from tpcp import Algorithm, Parameter, OptimizableParameter, make_action_safe

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

class BiLSTM(Algorithm):
    _action_methods = "predict"

    # INPUT PARAMETERS
    # 1. Model architecture
    input_shape: tuple
    bi_lstm_units: OptimizableParameter[int]
    first_dropout_rate: OptimizableParameter[float]
    mono_lstm_units: OptimizableParameter[int]
    second_dropout_rate: OptimizableParameter[float]
    dense_layer_units: OptimizableParameter[int]
    # 2. Hyperparamters
    learning_rate:  OptimizableParameter[float]
    num_epochs: OptimizableParameter[int]
    batch_size: OptimizableParameter[int]

    # THE MODEL
    _model: Optional[keras.Sequential]

    # Results
    predictions_: pd.Series

    def __init__(
        self,
        input_shape,
        bi_lstm_units: int = 64,
        first_dropout_rate: float = 0.6,
        mono_lstm_units: int = 128,
        second_dropout_rate: float = 0.6,
        dense_layer_units: int = 400,
        learning_rate:  float = 0.001,
        num_epochs: int = 12,
        batch_size: int = 128,
        _model = None
    ):
        self.input_shape = input_shape
        self.bi_lstm_units = bi_lstm_units
        self.first_dropout_rate = first_dropout_rate
        self.mono_lstm_units = mono_lstm_units
        self.second_dropout_rate = second_dropout_rate
        self.dense_layer_units = dense_layer_units
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self._model = _model

        
    def self_optimize(self, training_data: np.ndarray, labels: np.ndarray):
        """Use the training data and the corresponding labels to train the model with the hyperparameters passed in the init

        Args:
            training_data (np.ndarray): training data, multiple inputs
            labels (np.ndarray): corresponding labels

        Returns:
            self: with _model now set to a trained model
        """

        if self._model==None:
            self._create_model()

        assert training_data.shape[1] == self.input_shape[0], f"You are trying to train the model with input of the wrong dimensions. The required input shape is {self.input_shape} and your provided training_data has shape {training_data.shape}!"
        assert training_data.shape[2] == self.input_shape[1], f"You are trying to train the model with input of the wrong dimensions. The required input shape is {self.input_shape} and your provided training_data has shape {training_data.shape}!"

        self._model.fit(training_data, labels,
          epochs=self.num_epochs, 
          batch_size=self.batch_size,
          validation_split=0.1,
          shuffle=True)
        
        return self
    

    @make_action_safe
    def predict(self, input_data: np.ndarray):
        """Accepts a single data point and returns the prediction of the trained network for it

        Args:
            input_data (np.ndarray): Single input to be labeled by the trained network

        Returns:
            np.ndarray: Prediction of the model
        """
        
        assert self._model != None, "Before making a prediction you will have to train a model using the self.self_optimize method of the same instance!"

        self.predictions_ = np.squeeze(self._model.predict(input_data))
        
        return self
    
    
    def _create_model(self):
        """Helper function generating a new instance of the BiLSTM model
        """
        self._model = keras.Sequential(
            [
                layers.Input(shape=self.input_shape),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.bi_lstm_units,return_sequences=True)),
                tf.keras.layers.Dropout(self.first_dropout_rate),
                tf.keras.layers.LSTM(self.mono_lstm_units),
                tf.keras.layers.Dropout(self.second_dropout_rate),
                tf.keras.layers.Dense(self.dense_layer_units)
            ]
        )
        self._model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")

        return self
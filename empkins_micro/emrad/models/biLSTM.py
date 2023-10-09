from typing import Optional
from empkins_io.datasets.d03.micro_gapvii._dataset import MicroBaseDataset

from tpcp import Algorithm, Parameter, OptimizableParameter, make_action_safe

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

class BiLSTM(Algorithm):
    """Algorithm to train a bi-directional LSTM Network with layers [biLSTM, Dropout, monoLSTM, Dropout, Dense] and also make predictions after training.

    Inheriting from:
        tpcp.Algorithm: Base-class for tpcp algorithm objects.
    """
    
    _action_methods = "predict"

    # INPUT PARAMETERS
    # 1. Model architecture
    bi_lstm_units: OptimizableParameter[int]
    first_dropout_rate: OptimizableParameter[float]
    mono_lstm_units: OptimizableParameter[int]
    second_dropout_rate: OptimizableParameter[float]
    # 2. Hyperparamters
    learning_rate:  OptimizableParameter[float]
    num_epochs: OptimizableParameter[int]
    batch_size: OptimizableParameter[int]

    # THE MODEL
    _model: Optional[keras.Sequential]

    # Results
    predictions_: np.ndarray

    def __init__(
        self,
        bi_lstm_units: int = 64,
        first_dropout_rate: float = 0.6,
        mono_lstm_units: int = 128,
        second_dropout_rate: float = 0.6,
        learning_rate:  float = 0.001,
        num_epochs: int = 12,
        batch_size: int = 128,
        _model = None
    ):
        self.bi_lstm_units = bi_lstm_units
        self.first_dropout_rate = first_dropout_rate
        self.mono_lstm_units = mono_lstm_units
        self.second_dropout_rate = second_dropout_rate
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
            self._create_model(training_data.shape[1], training_data.shape[2])

        assert self._model.layers[0].input_shape[1] == training_data.shape[1], f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        assert self._model.layers[0].input_shape[2] == training_data.shape[2], f"Your training data has dimension {training_data.shape} while the model has input shape {self._model.layers[0].input_shape}!"
        
        self._model.fit(training_data, labels,
          epochs=self.num_epochs, 
          batch_size=self.batch_size,
          validation_split=0.1,
          shuffle=True)
        
        return self
    

    #@make_action_safe
    def predict(self, input_data: np.ndarray):
        """Accepts a single data point and returns the prediction of the trained network for it

        Args:
            input_data (np.ndarray): Single input to be labeled by the trained network

        Returns:
            np.ndarray: Prediction of the model
        """
        
        assert self._model != None, "Before making a prediction you will have to train a model using the self.self_optimize method of the same instance!"

        #model_copy = keras.models.clone_model(self._model)

        self.predictions_ = np.squeeze(self._model.predict(input_data))
        
        return self
    
    
    def _create_model(self, timesteps_per_sample: int, num_features: int):
        """Helper function generating a new instance of the BiLSTM model
        """
        self._model = keras.Sequential(
            [
                layers.Input(shape= (timesteps_per_sample, num_features)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.bi_lstm_units,return_sequences=True)),
                tf.keras.layers.Dropout(self.first_dropout_rate),
                tf.keras.layers.LSTM(self.mono_lstm_units),
                tf.keras.layers.Dropout(self.second_dropout_rate),
                tf.keras.layers.Dense(timesteps_per_sample)
            ]
        )
        self._model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")

        return self
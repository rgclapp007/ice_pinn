from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Initializer
from ._constants import *

class TunableXavierNormal(Initializer):
    """
    Custom Xavier normal initializer with a tunable scaling factor.

    Attributes:
        lsnow (float): Scaling factor for the standard deviation.
    """

    def __init__(self, lsnow: float):
        self._lsnow = float(lsnow)

    def __call__(self, shape, dtype=tf.float32):
        xavier_stddev = tf.sqrt(2 / tf.math.reduce_sum(shape)) * self._lsnow
        return tf.Variable(tf.random.truncated_normal(shape, stddev=xavier_stddev, dtype=dtype))

    def get_config(self):
        return {"lsnow": self._lsnow}


class ShiftLayer(Layer):
    """
    A custom layer that linearly shifts and scales the input data.

    """

    def __init__(self,lx,ux):
        super().__init__()
        self._lx=lx
        self._ux=ux
        
    def call(self, inputs):
        return 2.0 * (inputs - self._lx) / (self._ux - self._lx) - 1.0


def create_mlp(layers: List[int], lyscl: List[float], dtype=tf.float64):
    """
    Creates a multilayer perceptron (MLP) model for PINN problems.

    Args:
        layers (List[int]): Number of units in each layer of the MLP.
        lyscl (List[float]): Scaling factors for the Xavier initializer for each layer.
        dtype (tf.dtype): Data type of the input and layers.

    Returns:
        tf.keras.Model: A compiled Keras model of the MLP.
    """

    inputs = Input(shape=(1,), dtype=dtype)
    shifted = ShiftLayer(0,1)(inputs)
    dense = Dense(
        layers[0], activation="tanh", dtype=dtype,
        kernel_initializer=TunableXavierNormal(lyscl[0]))(shifted)
    for n_unit, stddev in zip(layers[1:-1], lyscl[1:-1]):
        dense = Dense(
            n_unit, activation="tanh", dtype=dtype,
            kernel_initializer=TunableXavierNormal(stddev))(dense)
    dense = Dense(
        layers[-1], activation=None, dtype=dtype,
        kernel_initializer=TunableXavierNormal(lyscl[-1]))(dense)
    model = Model(inputs=inputs, outputs=dense)
    return model


import os
import math
import json
import jsonpickle

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Layer, Dense, Concatenate, BatchNormalization, LeakyReLU
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras import __version__ as keras_version

from ntopo.utils import transform_minus11, load_file

class IdentityFeatures(Layer):
    def __init__(self, n_input):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_input

    def get_config(self):
        return {'n_input': self.n_input}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return inputs


class ConcatSquareFeatures(Layer):
    def __init__(self, n_input):
        super().__init__()

        self.n_input = n_input
        self.n_output = 2*n_input

    def get_config(self):
        return {'n_input': self.n_input}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        out = tf.concat((inputs, inputs*inputs), axis=-1)
        return out


class ConcatSineFeatures(Layer):
    def __init__(self, n_input):
        super().__init__()

        self.n_input = n_input
        self.n_output = 2*n_input

    def get_config(self):
        return {'n_input': self.n_input}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        out = tf.concat((inputs, tf.math.sin(inputs)), axis=-1)
        return out


class DenseSIRENModel(Model):
    def __init__(self, n_input, n_output, n_hidden, last_layer_init_scale, omega0, use_omega_split = False):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.last_layer_init_scale = last_layer_init_scale
        self.omega0 = omega0

        sin_activation_first_layer = lambda x: K.sin(omega0*x)
        first_layer_initializer = tf.keras.initializers.RandomUniform(
            minval=-1.0/n_input, maxval=1.0/n_input)
        k_sqrt_first = np.sqrt(1.0/n_input)
        bias_initializer_first = tf.keras.initializers.RandomUniform(
            minval=-k_sqrt_first, maxval=k_sqrt_first)

        if use_omega_split:
            sin_activation = lambda x: K.sin(omega0*x)
            w_middle_deviation = np.sqrt(6.0 / n_hidden) / omega0
            k_sqrt_middle = np.sqrt(1.0/n_hidden) / omega0
        else:
            sin_activation = lambda x: K.sin(x)
            w_middle_deviation = np.sqrt(6.0 / n_hidden)
            k_sqrt_middle = np.sqrt(1.0/n_hidden)

        weight_initializer_middle_layer = tf.keras.initializers.RandomUniform(
            minval=-w_middle_deviation, maxval=w_middle_deviation)
        bias_initializer = tf.keras.initializers.RandomUniform(
            minval=-k_sqrt_middle, maxval=k_sqrt_middle)

        last_initializer = tf.keras.initializers.RandomUniform(
            minval=-np.sqrt(6 / n_hidden) * last_layer_init_scale, maxval=np.sqrt(6 / n_hidden) * last_layer_init_scale)

        self.dense0 = Dense(n_hidden, activation=sin_activation_first_layer, kernel_initializer=first_layer_initializer,
                            bias_initializer=bias_initializer_first)
        self.dense1 = Dense(n_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer,
                            bias_initializer=bias_initializer)
        self.dense2 = Dense(n_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer,
                            bias_initializer=bias_initializer)
        self.dense3 = Dense(n_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer,
                            bias_initializer=bias_initializer)
        self.dense4 = Dense(n_hidden, activation=sin_activation, kernel_initializer=weight_initializer_middle_layer,
                            bias_initializer=bias_initializer)
        self.dense5 = Dense(n_output, kernel_initializer=last_initializer)

    def get_config(self):
        config = {
            'n_input': self.n_input,
            'n_output': self.n_output,
            'n_hidden': self.n_hidden,
            'last_layer_init_scale': self.last_layer_init_scale,
            'omega0': self.omega0
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        l0 = self.dense0(inputs)
        l1 = self.dense1(l0)
        l2 = self.dense2(l1)
        l2_concate = Concatenate()([inputs, l2])
        l3 = self.dense3(l2_concate)
        l4 = self.dense4(l3)
        l4_concate = Concatenate()([l2_concate, l4])
        l5 = self.dense5(l4_concate)
        return l5


class FCBNLeakyReluModel(Model):

    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden

        self.model = Sequential(
            [
                Input(n_input),
                BatchNormalization(),
                Dense(n_hidden),
                LeakyReLU(),
                BatchNormalization(),
                Dense(n_hidden),
                LeakyReLU(),
                BatchNormalization(),
                Dense(n_hidden),
                LeakyReLU(),
                BatchNormalization(),
                Dense(n_hidden),
                LeakyReLU(),
                BatchNormalization(),
                Dense(n_hidden),
                LeakyReLU(),
                BatchNormalization(),
                Dense(n_output)
            ]
        )

    def get_config(self):
        return {
            'n_input': self.n_input,
            'n_output': self.n_output,
            'n_hidden': self.n_hidden,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return self.model(inputs)


class ModelPartialQ(Model):
    def __init__(self, model, q):
        super().__init__()

        self.model = model
        assert isinstance(q, np.ndarray), 'type of q is ' + str(type(q))
        assert q.shape[0] == 1 and q.shape[1] == 1
        self.q = tf.constant(q, dtype=tf.float32)

    def call(self, inputs):
        inputs_with_q = tf.concat(
            (inputs, tf.tile(self.q, multiples=tf.stack((tf.shape(inputs)[0], 1)))), axis=1)
        return self.model(inputs_with_q)


class DispModel(Model):

    def __init__(
        self,
        input_domain,
        dim,
        bc,
        features=None,
        model=None,
    ):
        super().__init__()

        self.input_domain = np.array(input_domain, dtype=np.float32)
        self.dim = dim

        if isinstance(bc, str):
            self.bc = jsonpickle.decode(bc)
        else:
            self.bc = bc

        self.features = getattr(
            globals()[features['class_name']], 'from_config')(features['config'])
        self.model = getattr(
            globals()[model['class_name']], 'from_config')(model['config'])

        assert self.features.n_output == self.model.n_input

        # call model with some stuff to initialize the shapes
        some_stuff = tf.ones((5, len(input_domain)//2))
        
        self.predict(some_stuff)

    def get_config(self):
        assert jsonpickle.encode(self.bc) != 'null', 'serializing bc failed'
        config = {
            'input_domain': self.input_domain,
            'dim': self.dim,
            'bc': jsonpickle.encode(self.bc),
            'features': {
                'class_name': self.features.__class__.__name__,
                'config': self.features.get_config(),
            },
            'model': {
                'class_name': self.model.__class__.__name__,
                'config': self.model.get_config(),
            },
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def load_from_config_file(cls, filename):
        return tf.keras.models.model_from_json(load_file(filename), custom_objects={cls.__name__: cls})

    def call(self, inputs, training=None):
        assert len(self.input_domain) // 2 == inputs.shape.as_list()[1]

        x = tf.gather(inputs, [0], axis=1)
        y = tf.gather(inputs, [1], axis=1)
        if self.dim == 3:
            z = tf.gather(inputs, [2], axis=1)

        inputs_m11 = transform_minus11(inputs, self.input_domain)

        features_output = self.features(inputs_m11)
        displacement = self.model(features_output)

        if self.dim == 2:
            bc_output = self.bc([x, y])
        else:
            assert self.dim == 3
            bc_output = self.bc([x, y, z])

        displacement_fixed = bc_output * displacement
        return displacement_fixed

    def get_model_partial_q(self, q):
        return ModelPartialQ(self, q)

    def has_q(self):
        return len(self.input_domain) // 2 > self.dim


class DensityModel(Model):
    def __init__(self,
                 input_domain,
                 dim,
                 volume_ratio,
                 constraint=None,
                 constraint_config=None,
                 features=None,
                 model=None,
                 volume_ratio_q_idx=-1
                 ):
        super().__init__()

        if constraint_config is None:
            assert constraint is not None
            self.constraint = constraint
        else:
            self.constraint = jsonpickle.decode(constraint_config)

        self.input_domain = np.array(input_domain, dtype=np.float32)
        self.dim = dim
        self.volume_ratio = volume_ratio
        self.volume_ratio_q_idx = volume_ratio_q_idx

        self.features = getattr(
            globals()[features['class_name']], 'from_config')(features['config'])
        self.model = getattr(
            globals()[model['class_name']], 'from_config')(model['config'])

        assert self.features.n_output == self.model.n_input

        # throw some stuff at the model to initialize the shapes
        self.predict(tf.ones((5, len(input_domain)//2)))

    def get_config(self):
        config = {
            'input_domain': self.input_domain,
            'dim': self.dim,
            'volume_ratio': self.volume_ratio,
            'volume_ratio_q_idx': self.volume_ratio_q_idx,
            'constraint_config': jsonpickle.encode(self.constraint),
            'features': {
                'class_name': self.features.__class__.__name__,
                'config': self.features.get_config(),
            },
            'model': {
                'class_name': self.model.__class__.__name__,
                'config': self.model.get_config(),
            }
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def load_from_config_file(cls, filename):
        return tf.keras.models.model_from_json(load_file(filename), custom_objects={cls.__name__: cls})

    def call(self, inputs, training=None):
        assert len(self.input_domain) // 2 == inputs.shape.as_list()[1]

        if self.volume_ratio_q_idx == -1:
            volume_ratio = self.volume_ratio
        else:
            volume_ratio = tf.gather(
                inputs, [self.dim + self.volume_ratio_q_idx], axis=1)

        inputs_m11 = transform_minus11(inputs, self.input_domain)

        inputs_features = self.features(inputs_m11)
        model_output = self.model(inputs_features)

        # sigmoid is y = 1/(1 + exp(-x))
        # we want that initialization sigmoid(0 + offset) = volume_ratio, inverse of sigmoid x = ln(y/(1-y))

        alpha = 5.0
        offset = tf.math.log(volume_ratio / (1.0 - volume_ratio))
        densities = tf.math.sigmoid(alpha * model_output + offset)

        densities_constrained = self.constraint.apply(inputs, densities)

        return densities_constrained

    def get_model_partial_q(self, inputs):
        return ModelPartialQ(self, inputs)


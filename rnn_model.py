from tensorflow import math
import tensorflow as tf
import numpy as np


class WaveModule(tf.Module):
    def __init__(self, dt, b, Nx, Ny):
        """Analog to the RNN"""
        super().__init__()
        self.dt = tf.constant(dt, name='dt')
        c = np.zeros([Nx, Ny])
        self.c = tf.Variable(c, name='c', trainable=True)
        self.b = tf.constant(b, name='b')
        self.laplacian = tf.reshape(tf.constant([[0.0,  1.0,  0.0],
                                                 [1.0, -4.0,  1.0],
                                                 [0.0,  1.0,  0.0]],
                                                dtype='float32'),
                                    [3, 3, 1, 1])

    def time_step(self, x, y1, y2):
        """Take a step through time.

        Parameters
        ----------
        x : Input value(s) at current time step, batched in first dimension
        y1 : Scalar wave field one time step ago (part of the hidden state)
        y2 : Scalar wave field two time steps ago (part of the hidden state)
        """
        dt = self.dt
        c = self.c
        b = self.b

        term_one = 2 + dt**(-2)*math.multiply(c.pow(2),
                                              tf.squeeze(tf.nn.conv2d(
                                                  tf.expand_dims(y1, 3),
                                                  self.laplacian,
                                                  1,
                                                  "SAME"),
                                                  1))

        term_two = math.multiply(-1 - dt * b, y2)
        denominator = dt ** (-2) + b * 0.5 * dt ** (-1)
        y = math.multiply(denominator.pow(-1),
                          math.add(term_one, term_two))

        # Insert the source
        y_out = y[:, self.src_x, self.src_y]
        y_out = y_out + tf.broadcast_to(x, tf.shape(y_out))

        return y_out, y1

    def __call__(self, x, probe_output=True):
        """Propagate forward in time for the length of the input.

        Parameters
        ----------
        x :
            Input sequence(s), batched in first dimension
        probe_output : bool
            Defines whether the output is the probe vector or the entire spatial
            distribution of the scalar wave field in time
        """
        # hacky way of figuring out if we're on the GPU from inside the model
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        # First dim is batch
        batch_size = x.shape[0]

        # init hidden states
        y1 = tf.zeros([batch_size, self.Nx, self.Ny], dtype=tf.dtypes.float32)
        y2 = tf.zeros([batch_size, self.Nx, self.Ny], dtype=tf.dtypes.float32)
        y_all = []

        for xi in x:
            y, y1 = self.time_step(xi, y1, y2)
            y_all.append(y)

        y = tf.stack(y_all, axis=1)

        return y


def sat_damp(u, uth=1.0, b0=1.0):
    return b0 / (1 + math.abs(u/uth).pow(2))

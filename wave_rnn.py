from tensorflow import math
import tensorflow as tf
import numpy as np

LAP = [[0.0,  1.0,  0.0],
       [1.0, -4.0,  1.0],
       [0.0,  1.0,  0.0]]


class WaveLayer(tf.keras.layers.Layer):
    def __init__(self, dt, h, b, Nx, Ny):
        """Class to compute iterations of the wave equation

        Args:
            dt: computational time step size
            h: computational spatial step size
            b: damping parameter
            Nx : number of x-direction grid cells in the computational domain
            Ny : number of y-direction grid cells in the computational domain
        """
        super().__init__()
        self.dt = tf.constant(dt, name='dt')
        self.Nx = Nx
        self.Ny = Ny
        c = np.ones([Nx, Ny])
        self.c = tf.Variable(c, name='c', trainable=True)
        self.b = tf.constant(b, name='b')

        self.laplacian = tf.constant(LAP, dtype='float32') / h**2
        # reshape to be in the format expected by the conv2d operator
        self.laplacian = tf.reshape(self.laplacian, [3, 3, 1, 1])

    def compute_laplacian(self, field):
        """Compute  the laplacian of the given field. Uses the conv2d operator
        which expects the filter to be in format: (filter_height, filter_width,
        in_width, in_channels)

        Args:
            field: Scalar wave field, expected to be in the shape of
            (batch_n, in_height, in_width).
        """
        out = tf.nn.conv2d(tf.expand_dims(field, 3), self.laplacian, 1, "SAME")
        return tf.squeeze(out, 3)

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

        term_a = 2 + dt**2*math.multiply(c.pow(2), self.compute_laplacian(y1))

        term_two = math.multiply(-1 - dt * b, y2)
        denominator = dt ** (-2) + b * 0.5 * dt ** (-1)
        y = math.multiply(denominator.pow(-1),
                          math.add(term_a, term_two))

        # Insert the source
        y_out = y[:, self.src_x, self.src_y]
        y_out = y_out + tf.broadcast_to(x, tf.shape(y_out))

        return y_out, y1

    def call(self, x, probes):
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
        total_sum = 0
        y_outs = []
        for probe_crd in probes:
            px, py = probe_crd
            y_out = math.reduce_sum(math.square(y[:,:,px,py]))
            total_sum += y_out
            y_outs.append(y_out)

        y_outs = tf.constant(y_outs) / total_sum

        return y_outs


class WaveModel(tf.keras.Model):
    def __init__(self, dt, h, b, Nx, Ny):
        super().__init__()
        self.wave_layer = WaveLayer(dt, h, b, Nx, Ny)

    def call(self, input, probes):
        return self.wave_layer(input, probes)


def sat_damp(u, uth=1.0, b0=1.0):
    return b0 / (1 + math.abs(u/uth).pow(2))

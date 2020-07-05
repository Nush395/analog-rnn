from tensorflow import math
import tensorflow as tf
import numpy as np

LAP = [[0.0,  1.0,  0.0],
       [1.0, -4.0,  1.0],
       [0.0,  1.0,  0.0]]


class WaveLayer(tf.keras.layers.Layer):
    def __init__(self, dt, h, Nx, Ny):
        """Class to compute iterations of the wave equation

        Args:
            dt: computational time step size
            h: computational spatial step size
            Nx : number of x-direction grid cells in the computational domain
            Ny : number of y-direction grid cells in the computational domain
        """
        super().__init__()
        self.dt = tf.constant(dt, name='dt')
        c = np.ones([Nx, Ny])
        self.c_lin = tf.constant(c, name='c_lin')
        self.c_nl = self.add_weight(shape=(Nx, Ny),
                                    initializer='random_normal',
                                    trainable=True)
        # used to calculate c = c + u^2*c_nl

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
        y1 : Scalar wave field u(x,y) one time step ago (part of the hidden state)
        y2 : Scalar wave field u(x,y) two time steps ago (part of the hidden state)
        """
        dt = self.dt
        # c(x,y) = c_lin(x,y) + u_t(x,y)^2 * c_nl(x,y)
        c = math.add(self.c_lin,
                     math.multiply(math.square(y1), self.c_nl))

        y = math.add(tf.constant(2),
                     math.multiply(math.multiply(math.square(dt),
                                                 math.square(c)),
                                   self.compute_laplacian(y1)))
        y = math.subtract(y, y2)

        # Insert the source
        src = tf.zeros(y.shape)
        src[self.src_x, self.src_y] = x
        y = math.add(y, math.multiply(math.square(dt), src))

        return y, y1

    def call(self, inputs, **kwargs):
        """Propagate forward in time for the length of the input.

        Args:
            inputs : Input sequence, batched in first dimension
        """
        # First dim is batch
        batch_size = inputs.shape[0]

        # init hidden states
        y = tf.zeros([batch_size, self.Nx, self.Ny], dtype=tf.dtypes.float32)
        y1 = tf.zeros([batch_size, self.Nx, self.Ny], dtype=tf.dtypes.float32)
        y_all = []

        # iterate sequence through time and collect output
        for xi in inputs:
            y, y1 = self.time_step(xi, y, y1)
            y_all.append(y)

        y = tf.stack(y_all, axis=1)
        total_sum = 0
        y_outs = []
        for probe_crd in self.probes:
            px, py = probe_crd
            y_out = math.reduce_sum(math.square(y[:,:,px,py]))
            total_sum += y_out
            y_outs.append(y_out)

        y_outs = tf.constant(y_outs) / total_sum

        return y_outs


class WaveModel(tf.keras.Model):
    def __init__(self, dt, h, Nx, Ny):
        super().__init__()
        self.wave_layer = WaveLayer(dt, h, Nx, Ny)

    def call(self, input, **kwargs):
        return self.wave_layer(input)


import tensorflow as tf

# might have to make a subclass of layer
class WaveSource:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def forward(self, Y, X, dt=1.0):
		X_expanded = tf.zeros(tf.shape(Y))
		X_expanded[:, self.x, self.y] = X

		return Y + dt ** 2 * X_expanded
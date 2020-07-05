from wave_rnn import WaveLayer
import unittest
import tensorflow as tf


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.wm = WaveLayer(1, 1, 100, 100)

    def test_laplacian_computation(self):
        # given
        field = tf.ones((1, 100, 100))

        # when
        result = self.wm.compute_laplacian(field)

        # then we expect these values at corners, edges and middle respectively
        self.assertEqual(result[0][0][0], -2)
        self.assertEqual(result[0][50][50], 0)
        self.assertEqual(result[0][99][50], -1)





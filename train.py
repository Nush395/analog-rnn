import tensorflow as tf
from rnn_model import model
import time
from main import load_all_audio


input_dir = ""
data = load_all_audio(input_dir)


loss_func = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()


def train_loop(input, ground_truth):
    with tf.GradientTape() as tape:
        outputs = model(input)
        loss = loss_func(ground_truth, outputs)
    gradients = tape.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_model(data):
    start = time.time()
    for epoch in range(5):
        for step, (x,y) in enumerate(data):
            loss = train_loop(x, y)
            print(loss)
        print("Epoch %d took %.4f s" % (epoch, time.time()-start))





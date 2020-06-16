import tensorflow as tf
from tensorflow import keras
from wave_rnn import WaveModel
import time
from data import load_datasets


train_ds, test_ds = load_datasets()


def assemble_callbacks():
    callbacks_list = []
    save_dir = 'checkpoints'
    callbacks_list.append(keras.callbacks.EarlyStopping(patience=4, verbose=1))
    callbacks_list.append(keras.callbacks.ModelCheckpoint(save_dir,
                                                          save_best_only=True,
                                                          verbose=1))
    callbacks_list.append(keras.callbacks.TensorBoard(save_dir))
    return callbacks_list


def train(input, ground_truth):
    b = np.ones()
    model = WaveModel(1, 1, b, 201, 201)
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = assemble_callbacks()
    model.fit(x=input, y=ground_truth, callbacks=callbacks, epochs=10,
              validation_split=0.2)


def train_loop(input, ground_truth):
    optimizer = keras.optimizers.Adam()
    loss_func = keras.losses.BinaryCrossentropy()

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





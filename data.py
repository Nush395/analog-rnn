from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(files, labels):
    file_path_ds = tf.data.Dataset.from_tensor_slices(files)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((file_path_ds, label_ds))


def load_audio(file_path, label):
    audio = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio,
                                             desired_channels=1,
                                             desired_samples=16000)
    return audio, label


def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=64):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    # Repeat dataset forever
    ds = ds.repeat()
    # Prepare batches
    ds = ds.batch(batch_size)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def load_datasets(path='vowels'):
    files = os.listdir(path)
    labels = [file.split('.')[0][-2] for file in files]

    # split the available data by file into train and test
    train_files, test_files, train_labels, test_labels = \
        train_test_split(files, labels, test_size=0.33, random_state=42)

    train_ds = get_dataset(train_files, train_labels)
    train_ds = prepare_for_training(train_ds)
    test_ds = get_dataset(test_files, test_labels)
    test_ds = prepare_for_training(test_ds)

    return train_ds, test_ds

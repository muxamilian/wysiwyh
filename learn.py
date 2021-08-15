import numpy as np
import tensorflow as tf
from datetime import datetime
import math
import os

from tensorflow.keras.callbacks import Callback
from model import Autoencoder, process_img

from glob import glob

img_size = 64
batch_size = 64

x_files = glob('data/*.jpg')

files_ds = tf.data.Dataset.from_tensor_slices(x_files)

raw_ds = files_ds.map(lambda x: process_img(x, img_size)).cache()
train_ds = raw_ds.shuffle(10000,reshuffle_each_iteration=True).batch(batch_size)
val_ds = raw_ds.shuffle(10000,reshuffle_each_iteration=False, seed=0).take(batch_size).batch(batch_size)
val_batch = next(iter(val_ds))

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
  tf.summary.image("In imgs", val_batch, step=0)

training_data = train_ds.take(math.floor(len(raw_ds)/batch_size))

def create_image_from_distribution(sequence, n_bins=100):
  assert len(sequence.shape) == 1
  output_image = np.zeros((n_bins, sequence.shape[-1], 3), dtype=np.float32)
  for i in range(len(sequence)):
    output_image[:int(np.floor(n_bins*sequence[i])), i, :] = 1.0

  return output_image

class CustomCallback(Callback):

  def on_epoch_end(self, epoch, logs=None):
    out_imgs, out_codes = autoencoder(val_batch)

    with file_writer.as_default():
      tf.summary.image("Out imgs", out_imgs, step=epoch)
      tf.summary.histogram("Out overall dists", out_codes, step=epoch)
      stacked_dist_images = np.stack([
        create_image_from_distribution(out_codes[0,:].numpy()), 
        create_image_from_distribution(out_codes[1,:].numpy()),
        create_image_from_distribution(out_codes[2,:].numpy())])
      tf.summary.image("Out dists", stacked_dist_images, step=epoch)

      for key in logs:
        tf.summary.scalar(key, logs[key], step=epoch)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

optimizer = tf.keras.optimizers.Adam()

autoencoder = Autoencoder(100, 5, batch_size, img_size)
autoencoder.compile(optimizer=optimizer, run_eagerly=True)

autoencoder.fit(training_data,
                epochs=1000,
                callbacks=[
                  CustomCallback(), 
                  tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(logdir, "weights.{epoch:02d}-{total_loss:.5f}"), monitor='total_loss', verbose=1, save_best_only=False,
                    save_weights_only=False, mode='min', save_freq=1000
                  )],
                shuffle=False)

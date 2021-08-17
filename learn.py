import numpy as np
import tensorflow as tf
from datetime import datetime
import math
import os
import random

from model import Autoencoder, process_img, CustomCallback

from glob import glob

img_size = 64
batch_size = 64
train = False

x_files = glob('data/*.jpg')

files_ds = tf.data.Dataset.from_tensor_slices(x_files)

raw_ds = files_ds.map(lambda x: process_img(x, img_size)).cache()

optimizer = tf.keras.optimizers.Adam()

autoencoder = Autoencoder(100, 5, batch_size, img_size)
autoencoder.compile(optimizer=optimizer, run_eagerly=True)

# def randomize_phase(absolute_value):
#   absolute_value_squared = absolute_value**2
#   real_part_squared = np.random.uniform(low=0, high=absolute_value_squared)
#   imag_part_squared = absolute_value_squared - real_part_squared
#   return np.sqrt(real_part_squared) + 1j * np.sqrt(imag_part_squared)

def randomize_phase(absolute_values):
  random_angles = np.random.uniform(low=0, high=np.array(2*np.pi).repeat(len(absolute_values)))
  real_part = absolute_values * np.cos(random_angles)
  imag_part = absolute_values * np.sin(random_angles)
  complex_results = real_part + 1j * imag_part
  assert np.isclose(np.abs(complex_results), absolute_values).all()
  return complex_results

if train:

  train_ds = raw_ds.shuffle(10000,reshuffle_each_iteration=True).batch(batch_size)
  training_data = train_ds.take(math.floor(len(raw_ds)/batch_size))
  val_ds = raw_ds.shuffle(10000,reshuffle_each_iteration=False, seed=0).take(batch_size).batch(batch_size)
  val_batch = next(iter(val_ds))

  logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  file_writer = tf.summary.create_file_writer(logdir)

  with file_writer.as_default():
    tf.summary.image("In imgs", val_batch, step=0)

  autoencoder.fit(training_data,
                  epochs=1000,
                  callbacks=[
                    CustomCallback(file_writer, val_batch), 
                    tf.keras.callbacks.ModelCheckpoint(
                      os.path.join(logdir, "weights.{epoch:02d}-{total_loss:.5f}"), monitor='total_loss', verbose=1, save_best_only=False,
                      save_weights_only=False, mode='min', save_freq=1000
                    )],
                  shuffle=False)

else:
  autoencoder.load_weights('logs/20210815-014755/weights.1000-0.00602/variables/variables')
  predict_ds = raw_ds.batch(batch_size).take(1)
  _, predicted_code = autoencoder.predict(predict_ds)

  first_code = predicted_code[0,:].astype(np.float64)
  # first_code /= len(first_code)
  first_code = np.concatenate(([0,0], first_code))
  reps = 49
  first_code = first_code.repeat(reps)
  first_code = np.concatenate((np.zeros(5000-len(first_code)+1), first_code))
  # first_code /= reps

  time_series = np.fft.irfft(randomize_phase(first_code))

  pass
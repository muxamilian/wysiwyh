import numpy as np
import tensorflow as tf
from datetime import datetime
import math
import os
import pyaudio
import matplotlib.pyplot as plt

from model import Autoencoder, process_img, CustomCallback

from glob import glob

img_size = 64
batch_size = 64
train = False

optimizer = tf.keras.optimizers.Adam()

autoencoder = Autoencoder(100, 5, batch_size, img_size)
autoencoder.compile(optimizer=optimizer, run_eagerly=True)

# def randomize_phase(absolute_value):
#   absolute_value_squared = absolute_value**2
#   real_part_squared = np.random.uniform(low=0, high=absolute_value_squared)
#   imag_part_squared = absolute_value_squared - real_part_squared
#   return np.sqrt(real_part_squared) + 1j * np.sqrt(imag_part_squared)

def randomize_phase(absolute_values):
  real_part = absolute_values * np.cos(random_angles)
  imag_part = absolute_values * np.sin(random_angles)
  complex_results = real_part + 1j * imag_part
  assert np.isclose(np.abs(complex_results), absolute_values).all()
  return complex_results

def play_audio(audio_stream):
  p = pyaudio.PyAudio()
    # for paFloat32 sample values must be in range [-1.0, 1.0]
  stream = p.open(format=pyaudio.paFloat32,
                  channels=1,
                  rate=10000,
                  output=True)

  # play. May repeat with different volume values (if done interactively) 
  stream.write(audio_stream.tobytes())

  stream.stop_stream()
  stream.close()

  p.terminate()

if train:

  x_files = sorted(glob('data2/*.jpg'))
  files_ds = tf.data.Dataset.from_tensor_slices(x_files)
  raw_ds = files_ds.map(lambda x: process_img(x, img_size)).cache()

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
  autoencoder.load_weights('logs/20210818-181200/weights.1000-0.00864/variables/variables')

  video_fps = 60
  fps = 1

  x_files = sorted(glob('data2/*.jpg'))
  x_files_at_fps = [item for item in x_files if int(item.split('/')[-1].split('.')[0])%(video_fps/fps) == 0]
  
  files_ds = tf.data.Dataset.from_tensor_slices(x_files)
  raw_ds = files_ds.map(lambda x: process_img(x, img_size)).cache()
  predict_ds = raw_ds.batch(batch_size)

  predicted_code = autoencoder.predict(predict_ds)

  upper_limit_hz = 5000

  random_angles = np.random.uniform(low=0, high=np.array(2*np.pi).repeat(upper_limit_hz/fps+1))

  # current_batch = next(predict_ds.__iter__())
  # os.makedirs("figures", exist_ok=True)
  # for i in range(current_batch.shape[0]):
  #   plt.close()
  #   plt.imshow(current_batch[i,...])
  #   plt.savefig(f"figures/img_{i}.png")

  final_time_series = []
  for i in range(predicted_code.shape[0]):
    current_code = predicted_code[i,:].astype(np.float64)
    # plt.close()
    # plt.plot(current_code)
    # plt.savefig(f"figures/code_{i}.pdf")

    current_code_repeated = current_code.repeat(int(math.floor(upper_limit_hz/current_code.shape[0]/fps)))
    current_code_filled_up = np.concatenate((np.zeros(1), current_code_repeated))

    # plt.close()
    # plt.plot(current_code_filled_up)
    # plt.savefig(f"figures/psd_{i}.pdf")
    time_series = np.fft.irfft(randomize_phase(current_code_filled_up)).astype(np.float32)
    # plt.close()
    # plt.plot(time_series)
    # plt.savefig(f"figures/time_series_{i}.pdf")
    final_time_series.append(time_series)

  complete_time_series = np.concatenate((*final_time_series,))
  first_five_time_series = np.concatenate((*final_time_series[:5],))
  # plt.close()
  # plt.plot(complete_time_series)
  # plt.savefig("figures/complete_time_series.pdf")
  # plt.close()
  # plt.plot(first_five_time_series)
  # plt.savefig("figures/first_five_time_series.pdf")
  volume = 1
  audio_stream = volume*complete_time_series

  play_audio(audio_stream)

  pass
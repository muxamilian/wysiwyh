from typing import final
import numpy as np
import tensorflow as tf
from datetime import datetime
import math
import os
import pyaudio
import matplotlib.pyplot as plt
import cv2
import time
from model import Autoencoder, process_img, convert_to_tf, CustomCallback
from glob import glob
import argparse

os.makedirs("figures", exist_ok=True)

parser = argparse.ArgumentParser(description="Either train a model, evaluate an existing one on a dataset or run live.")
parser.add_argument('--mode', type=str, default="live",
                    help='"train", "eval" or "live"')

args = parser.parse_args()
print("Got these arguments:", args)

img_size = 64
batch_size = 64
train = False

optimizer = tf.keras.optimizers.Adam()

autoencoder = Autoencoder(100, 7, batch_size, img_size)
autoencoder.compile(optimizer=optimizer, run_eagerly=True)

def randomize_phase(absolute_values):
  random_angles = np.random.uniform(low=0, high=np.array(2*np.pi).repeat(upper_limit_hz/fps+1))
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

if args.mode=='train':

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

elif args.mode=="eval":
  autoencoder.load_weights('logs/20210818-181200/weights.1000-0.00864/variables/variables')

  video_fps = 60
  fps = 10

  x_files = sorted(glob('data2/*.jpg'))
  x_files = [item for item in x_files if int(item.split('/image')[-1].split('.')[0])%(math.floor(video_fps/fps)) == 0]

  files_ds = tf.data.Dataset.from_tensor_slices(x_files)
  raw_ds = files_ds.map(lambda x: process_img(x, img_size)).cache()
  num_max_batches = math.floor(len(raw_ds)/batch_size)
  predict_ds = raw_ds.batch(batch_size).take(max(num_max_batches, 100))

  predicted_code = autoencoder.predict(predict_ds)

  upper_limit_hz = 5000

  # random_angles = np.random.uniform(low=0, high=np.array(2*np.pi).repeat(upper_limit_hz/fps+1))

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

elif args.mode=="live":
  # autoencoder.load_weights('logs/20210818-181200/weights.1000-0.00864/variables/variables')
  autoencoder.load_weights('logs/20210820-215243/weights.1000-0.00876/variables/variables')
  fps = 10

  upper_limit_hz = 5000
  volume = 1

  p = pyaudio.PyAudio()
    # for paFloat32 sample values must be in range [-1.0, 1.0]
  stream = p.open(format=pyaudio.paFloat32,
                  frames_per_buffer=int(2*upper_limit_hz/fps),
                  channels=1,
                  rate=2*upper_limit_hz,
                  output=True)

  cap = cv2.VideoCapture(0)

  i = -1
  while cap.isOpened():
      i += 1

      success, img = cap.read()

      if not success:
        continue

      start_time = time.time()

      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = img[0:720, 160:960, :]
      # plt.imshow(rgb_img),
      # plt.show()

      predict_ds = convert_to_tf(img, img_size)[None,:,:,:]

      predicted_code = autoencoder.predict(predict_ds)

      current_code = predicted_code[0,:].astype(np.float64)

      current_code_repeated = current_code.repeat(int(math.floor(upper_limit_hz/current_code.shape[0]/fps)))
      current_code_filled_up = np.concatenate((np.zeros(1), current_code_repeated))

      time_series = np.fft.irfft(randomize_phase(current_code_filled_up)).astype(np.float32)
      audio_to_be_played = time_series.tobytes()
      # final_time_series.append(time_series.tobytes())

      current_time = time.time()
      last_computation_duration = current_time - start_time

      write_start = time.time()
      stream.write(audio_to_be_played)
      writing_time = time.time()-write_start
      print("Writing time:", writing_time, "computation time:", last_computation_duration, "total:", writing_time+last_computation_duration)
      time.sleep(max(1/fps - 2*last_computation_duration, 0))

  stream.stop_stream()
  stream.close()

  p.terminate()

  pass
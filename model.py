import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import Callback
import tensorflow_probability as tfp

def create_image_from_distribution(sequence, n_bins=100):
  assert len(sequence.shape) == 1
  output_image = np.zeros((n_bins, sequence.shape[-1], 3), dtype=np.float32)
  for i in range(len(sequence)):
    output_image[n_bins-1-int(np.round(n_bins*sequence[i])):, i, :] = 1.0

  return output_image

class CustomCallback(Callback):

  def __init__(self, file_writer, val_batch):
    self.file_writer = file_writer
    self.val_batch = val_batch
    self.already_printed = False

  def on_batch_end(self, batch, logs):
    if not self.already_printed:
      print(self.model.encoder.summary())
      print(self.model.decoder.summary())
      self.already_printed = True

  def on_epoch_begin(self, epoch, logs):

    total_loss_tracker.reset_states()
    rec_loss_tracker.reset_states()
    code_loss_tracker.reset_states()
    corr_loss_tracker.reset_states()
    rmse_metric.reset_states()

  def on_epoch_end(self, epoch, logs=None):
    for i in range(self.model.bits_in_code+1):
      out_imgs, out_codes = self.model(self.val_batch, training=True, eval=True, level=i)

      distributions = [create_image_from_distribution(out_codes[i,:].numpy()) for i in range(8)]
      tag = f' {i}'

      with self.file_writer.as_default():
        tf.summary.image("Out imgs"+tag, out_imgs, step=epoch, max_outputs=8)
        tf.summary.histogram("Out overall dists"+tag, out_codes, step=epoch)
        stacked_dist_images = np.stack(distributions)
        tf.summary.image("Out dists"+tag, stacked_dist_images, step=epoch, max_outputs=8)

        for key in logs:
          tf.summary.scalar(key, logs[key], step=epoch)

def convert_to_tf(img, img_size):
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, size=(img_size, img_size), method=tf.image.ResizeMethod.AREA)
  return img

def process_img(file_path, img_size):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = convert_to_tf(img, img_size)
  return img

def get_triangle_distribution(half_size):
  dist = [1]
  step_size = 1/(half_size+1)
  for i in range(half_size):
    dist.append(1-(i+1)*step_size)
  dist = list(reversed(dist[1:])) + dist
  assert len(dist) == 2*half_size+1
  s = sum(dist)
  dist = [item/s for item in dist]
  return dist

loss_function = losses.MeanSquaredError()
total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
code_loss_tracker = tf.keras.metrics.Mean(name="code_loss")
corr_loss_tracker = tf.keras.metrics.Mean(name="corr_loss")
rmse_metric = tf.keras.metrics.RootMeanSquaredError()

class Autoencoder(Model):
  def __init__(self, code_dim, batch_size, img_size):
    super(Autoencoder, self).__init__()

    self.batch_size = batch_size
    self.img_size = img_size
    self.cdf = tf.cast(tf.linspace(0, 1, self.batch_size*code_dim+2)[1:-1], tf.float32)
    self.code_dim = code_dim

    bits_in_code = math.log2(self.code_dim)
    assert bits_in_code == round(bits_in_code)
    self.bits_in_code = int(bits_in_code)

    # self.smoothing_kernel = tf.constant(get_triangle_distribution(self.smoothing_half_size), dtype=tf.float32)[:,None,None]
    initial_size = 32

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(self.img_size, self.img_size, 3)),
      layers.Conv2D(initial_size, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(initial_size*2, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(initial_size*4, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(initial_size*8, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(initial_size*16, (4, 4), padding='same', strides=4),
      layers.LeakyReLU(alpha=0.1),
      layers.Flatten(),
      layers.Dense(initial_size*16),
      layers.LeakyReLU(alpha=0.1),
      # layers.Dense(self.code_dim+2*self.smoothing_half_size, activation=None),
      layers.Dense(self.code_dim, activation=None),
    ], name="encoder")

    self.decoder = tf.keras.Sequential([
      layers.Dense(initial_size*16),
      layers.LeakyReLU(alpha=0.1),
      layers.Reshape((1, 1, initial_size*16)),
      layers.Conv2DTranspose(initial_size*8, kernel_size=4, strides=4, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(initial_size*4, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(initial_size*2, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(initial_size, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(3, kernel_size=4, strides=2, activation='sigmoid', padding='same')
      ], name="decoder")

  def train_step(self, x):
    with tf.GradientTape() as tape:
      x_reconstructed, code = self(x, training=True, eval=False)
      reconstruction_loss = loss_function(x, x_reconstructed)
      # You won't believe it but sorting is broken for float32 and only returns 16 sorted values and then 0 from there on
      # reshaped_code = tf.cast(tf.reshape(code, (-1,)), dtype=tf.float64)
      # sorted = tf.cast(tf.sort(reshaped_code), dtype=tf.float32)
      reshaped_code = tf.reshape(code, (-1,))
      with tf.device('/cpu:0'):
        sorted = tf.sort(reshaped_code)

      brightness_of_each_image = tf.reduce_mean(x, axis=(1,2,3))
      avg_code = tf.reduce_mean(code, (1,))

      correlation = tfp.stats.correlation(brightness_of_each_image, avg_code, event_axis=None)

      deviation_loss = tf.reduce_mean((sorted-self.cdf)**2.)
      loss = reconstruction_loss + 0.01*deviation_loss - 0.01*correlation

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Compute our own metrics
    total_loss_tracker.update_state(loss)
    rec_loss_tracker.update_state(reconstruction_loss)
    code_loss_tracker.update_state(deviation_loss)
    corr_loss_tracker.update_state(correlation)
    rmse_metric.update_state(x, x_reconstructed)
    return {"total_loss": total_loss_tracker.result(), "rec_loss": rec_loss_tracker.result(), "code_loss": code_loss_tracker.result(), "corr_loss": corr_loss_tracker.result(), "rmse": rmse_metric.result()}

  def call(self, x, training=False, eval=True, level=-1):

    encoded = self.encoder(x, training=training and not eval)[:,:,None]

    if training and not eval or level != -1:
      encoded_list = tf.unstack(encoded)
      new_encoded = []
      for item in encoded_list:
        level_to_use = random.randint(0, self.bits_in_code) if level == -1 else level
        other_dim = int(2**level_to_use)

        item = split_at_level(item, other_dim)
        new_encoded.append(item)
      encoded = tf.stack(new_encoded)

    encoded = tf.reshape(encoded, (-1, self.code_dim))
    encoded = tf.sigmoid(encoded)

    if not training:
      return encoded

    decoded = self.decoder(encoded, training=training and not eval)
    return decoded, encoded

def split_at_level(item, level):
  code_dim = item.shape[0]
  item = tf.reshape(item, (-1, level))
  item = tf.reduce_mean(item, -1)
  item = tf.reshape(item, (-1, 1))
  item = tf.tile(item, (1, level))
  item = tf.reshape(item, (code_dim,))
  return item

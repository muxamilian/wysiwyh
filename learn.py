import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.python.keras.backend import reshape
from tensorflow.python.keras.preprocessing.image import smart_resize
from tensorflow.python.ops.gen_array_ops import Reshape
from tensorflow.keras.callbacks import Callback

img_size = 64
batch_size = 64

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

from glob import glob

x_files = glob('data/*.jpg')

files_ds = tf.data.Dataset.from_tensor_slices(x_files)

def process_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(img_size, img_size))
    return img

raw_ds = files_ds.map(lambda x: process_img(x)).cache()
train_ds = raw_ds.shuffle(10000,reshuffle_each_iteration=True).batch(batch_size)
val_ds = raw_ds.shuffle(10000,reshuffle_each_iteration=False, seed=0).take(batch_size).batch(batch_size)
val_batch = next(iter(val_ds))

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
  tf.summary.image("In imgs", val_batch, step=0)

# training_data = train_ds.map(lambda x: (x, x))
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

total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
code_loss_tracker = tf.keras.metrics.Mean(name="code_loss")
optimizer = tf.keras.optimizers.Adam()
loss_function = losses.MeanSquaredError()
rmse_metric = tf.keras.metrics.RootMeanSquaredError()

class Autoencoder(Model):
  def __init__(self, code_dim, smoothing_half_size):
    super(Autoencoder, self).__init__()

    self.cdf = tf.cast(tf.linspace(0, 1, batch_size*code_dim+2)[1:-1], tf.float32)
    self.code_dim = code_dim
    self.smoothing_half_size = smoothing_half_size

    self.smoothing_kernel = tf.constant(get_triangle_distribution(self.smoothing_half_size), dtype=tf.float32)[:,None,None]

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(img_size, img_size, 3)),
      layers.Conv2D(8, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(16, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(32, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(64, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2D(128, (4, 4), padding='same', strides=4),
      layers.LeakyReLU(alpha=0.1),
      layers.Flatten(),
      layers.Dense(256),
      layers.LeakyReLU(alpha=0.1),
      layers.Dense(self.code_dim+2*self.smoothing_half_size, activation=None),
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(256),
      layers.LeakyReLU(alpha=0.1),
      layers.Reshape ((  1, 1, 256)),
      layers.Conv2DTranspose(64, kernel_size=4, strides=4, padding='same'),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same'),
      layers.LeakyReLU(alpha=0.1),
      layers.Conv2DTranspose(3, kernel_size=4, strides=2, activation=None, padding='same')])

  def train_step(self, x):
    with tf.GradientTape() as tape:
      x_reconstructed, code = self(x, training=True)
      reconstruction_loss = loss_function(x, x_reconstructed)
      sorted = tf.sort(tf.reshape(code, (-1,)))
      deviation_loss = 0.1*tf.reduce_mean((sorted-self.cdf)**2.)
      loss = reconstruction_loss# + deviation_loss

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Compute our own metrics
    total_loss_tracker.update_state(loss)
    rec_loss_tracker.update_state(reconstruction_loss)
    code_loss_tracker.update_state(deviation_loss)
    rmse_metric.update_state(x, x_reconstructed)
    return {"total_loss": total_loss_tracker.result(), "rec_loss": rec_loss_tracker.result(), "code_loss": code_loss_tracker.result(), "rmse": rmse_metric.result()}

    @property
    def metrics(self):
        return [total_loss_tracker, rec_loss_tracker, code_loss_metric, rmse_metric]

  def call(self, x):
    encoded_unsmoothed = self.encoder(x)[:,:,None]
    
    smoothed = tf.nn.conv1d(encoded_unsmoothed, self.smoothing_kernel, stride=1, padding="VALID")
    smoothed = tf.reshape(smoothed, (batch_size, self.code_dim))
    encoded = tf.sigmoid(smoothed)

    decoded = self.decoder(encoded)
    return decoded, encoded

autoencoder = Autoencoder(100, 5)
autoencoder.compile(optimizer=optimizer, run_eagerly=True)

autoencoder.fit(training_data,
                epochs=1000,
                callbacks=[CustomCallback()],
                shuffle=False
                )

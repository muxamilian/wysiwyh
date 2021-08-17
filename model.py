import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import Callback

def create_image_from_distribution(sequence, n_bins=100):
  assert len(sequence.shape) == 1
  output_image = np.zeros((n_bins, sequence.shape[-1], 3), dtype=np.float32)
  for i in range(len(sequence)):
    output_image[:int(np.floor(n_bins*sequence[i])), i, :] = 1.0

  return output_image

class CustomCallback(Callback):

  def __init__(self, file_writer, val_batch):
    self.file_writer = file_writer
    self.val_batch = val_batch

  def on_epoch_end(self, epoch, logs=None):
    out_imgs, out_codes = self.model(self.val_batch)

    with self.file_writer.as_default():
      tf.summary.image("Out imgs", out_imgs, step=epoch)
      tf.summary.histogram("Out overall dists", out_codes, step=epoch)
      stacked_dist_images = np.stack([
        create_image_from_distribution(out_codes[0,:].numpy()), 
        create_image_from_distribution(out_codes[1,:].numpy()),
        create_image_from_distribution(out_codes[2,:].numpy())])
      tf.summary.image("Out dists", stacked_dist_images, step=epoch)

      for key in logs:
        tf.summary.scalar(key, logs[key], step=epoch)

def process_img(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(img_size, img_size))
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
rmse_metric = tf.keras.metrics.RootMeanSquaredError()

class Autoencoder(Model):
  def __init__(self, code_dim, smoothing_half_size, batch_size, img_size):
    super(Autoencoder, self).__init__()

    self.batch_size = batch_size
    self.img_size = img_size
    self.cdf = tf.cast(tf.linspace(0, 1, self.batch_size*code_dim+2)[1:-1], tf.float32)
    self.code_dim = code_dim
    self.smoothing_half_size = smoothing_half_size

    self.smoothing_kernel = tf.constant(get_triangle_distribution(self.smoothing_half_size), dtype=tf.float32)[:,None,None]

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(self.img_size, self.img_size, 3)),
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
      layers.Reshape ((1, 1, 256)),
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
      # You won't believe it but sorting is broken for float32 and only returns 16 sorted values and then 0 from there on
      reshaped_code = tf.cast(tf.reshape(code, (-1,)), dtype=tf.float64)
      sorted = tf.cast(tf.sort(reshaped_code), dtype=tf.float32)
      deviation_loss = tf.reduce_mean((sorted-self.cdf)**2.)
      loss = reconstruction_loss + 0.01*deviation_loss

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Compute our own metrics
    total_loss_tracker.update_state(loss)
    rec_loss_tracker.update_state(reconstruction_loss)
    code_loss_tracker.update_state(deviation_loss)
    rmse_metric.update_state(x, x_reconstructed)
    return {"total_loss": total_loss_tracker.result(), "rec_loss": rec_loss_tracker.result(), "code_loss": code_loss_tracker.result(), "rmse": rmse_metric.result()}

    # @property
    # def metrics(self):
    #     return [total_loss_tracker, rec_loss_tracker, code_loss_metric, rmse_metric]

  def call(self, x):
    encoded_unsmoothed = self.encoder(x)[:,:,None]
    
    smoothed = tf.nn.conv1d(encoded_unsmoothed, self.smoothing_kernel, stride=1, padding="VALID")
    smoothed = tf.reshape(smoothed, (self.batch_size, self.code_dim))
    encoded = tf.sigmoid(smoothed)

    decoded = self.decoder(encoded)
    return decoded, encoded

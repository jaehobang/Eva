import os
import cv2
import time
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,Reshape, UpSampling2D

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class CAENetwork:

  def __init__(self):

    self.autoencoder = None
    self.encoder = None
    self.width = 80  # requirements for this number must be that it needs to be divisible by 4
    self.height = 48  # must be divisible by 4
    self.image_array = None

  def _reform_input(self, image_table):
    # This function could rapidly change according to what we want
    # But the purpose of this function is to feed in the input as per network specifications
    # current specs: convert to size (48, 80) and grayscale
    start_time = time.time()
    assert(len(image_table.shape) == 4)
    assert(image_table.shape[3] == 3)
    n_samples = image_table.shape[0]
    self.image_array = np.ndarray(shape=(n_samples, self.height, self.width, 1))
    for i in range(len(image_table)):
      image = image_table[i ,: ,: ,:]
      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      image = cv2.resize(image, (self.width, self.height))
      # might need to expand dimension
      self.image_array[i ,: ,: ,0] = image

    self.image_array /= 255.0  # normalize the input for layers
    print("finished reforming input, total time taken is", time.time() - start_time, "seconds")
    return


  def _build(self):
    # TODO:
    # Combined network with both FC and CNN layers

    # Input
    input_img = Input(shape=(self.height, self.width, 1))
    # Encoder
    x = Conv2D(8 ,(3 ,3),
               activation='relu',
               padding='same')(input_img)
    x = Conv2D(8 ,(3 ,3),
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((2 ,2),
                     padding='same')(x)
    x = Conv2D(16 ,(3 ,3),
               activation='relu',
               padding='same')(x)
    x = Conv2D(16 ,(3 ,3),
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((2 ,2),
                     padding='same')(x) # Size
    x = Flatten()(x)
    encoded = Dense(256)(x)
    # Decoder
    x = Dense(16 *int(self.height / 4 ) *int(self.width / 4))(encoded)
    x = Reshape((int(self.height / 4), int(self.width / 4), 16))(x)
    x = UpSampling2D((2, 2))(x) # 24, 40, 16
    x = Conv2D(16, (3, 3),
               activation='relu',
               padding='same')(x)
    x = Conv2D(16, (3, 3),
               activation='relu',
               padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # Size 48, 80, 16
    x = Conv2D(8, (3, 3),
               activation='relu',
               padding='same')(x)
    decoded = Conv2D(1, (3, 3),
                     activation='relu',
                     padding='same')(x)

    self.autoencoder = Model(input_img, decoded)
    self.encoder = Model(input_img, encoded)
    self.autoencoder.compile(optimizer='adam', loss='mse')
    self.autoencoder.summary()



  def train(self, image_table):
    self._reform_input(image_table)
    self._build()
    n_samples = len(self.image_array)
    X_train = self.image_array[:int(n_samples * 0.8)]
    X_val = self.image_array[int(n_samples * 0.8):]

    start_time = time.time()
    train_history = self.autoencoder.fit(X_train, X_train, epochs=200, batch_size=2048, validation_data=(X_val, X_val))
    print("Total time it took to train autoencoder is ", time.time() - start_time, " seconds")

  def get_compressed(self, image_table):
    self._reform_input(image_table)
    images_compressed = self.encoder.predict(self.image_array)
    return images_compressed

  def get_features(self, image_table):
    # TODO - Ujjwal and Shreya
    pass


  def evaluate(self):
    # This function is currently unsupported
    pass


if __name__ == "__main__":
  cae = CAENetwork()

  from eva_storage.videoInputModule import VideoInputModule
  eva_dir = os.path.abspath('../')
  detrac_dir = os.path.join(eva_dir, 'data', 'ua_detrac', 'small-data')
  vim_ = VideoInputModule()
  vim_.convert_video(detrac_dir)
  image_table = vim_.get_image_array()
  cae.train(image_table)


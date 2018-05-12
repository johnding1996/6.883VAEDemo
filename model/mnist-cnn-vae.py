import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# pathname to save the model
KERAS_MODEL_FILEPATH = 'model.h5'

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 200
original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 0.01

x = Input(batch_shape=(batch_size,) + original_img_size)
conv_1 = Conv2D(img_chns, kernel_size=(2,2), padding='same', activation='relu')(x)
conv_2 = Conv2D(filters, kernel_size=(2,2), strides=(2,2), padding='same', activation='relu')(conv_1)
conv_3 = Conv2D(filters, kernel_size=num_conv, strides=(1,1), padding='same', activation='relu')(conv_2)
conv_4 = Conv2D(filters, kernel_size=num_conv, strides=(1,1), padding='same', activation='relu')(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 14 * 14, activation='relu')

decoder_reshape = Reshape((14, 14, filters))
decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, strides=(1,1),
                                   padding='same', activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, kernel_size=num_conv, strides=(1,1),
                                   padding='same', activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2),
                                          padding='valid', activation='relu')
decoder_mean_squash = Conv2D(img_chns, kernel_size=(2,2), padding='same', activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean_squash)
vae.compile(optimizer='adam', loss=vae_loss)

# number of epochs
epochs = 100

# train the VAE on MNIST digits
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=20)

vae.fit(x_train, x_train,
        validation_data=(x_test, x_test),
        shuffle=True, epochs=epochs, batch_size=batch_size, verbose=2,
        callbacks=[early_stopping])


# build a model to project inputs on the latent space# build
encoder = Model(x, z_mean)


batch_size = 1

_hid_decoded = Dense(intermediate_dim, activation='relu')
_up_decoded = Dense(filters * 14 * 14, activation='relu')
_reshape_decoded = Reshape((14, 14, filters))
_deconv_1_decoded = Conv2DTranspose(filters, kernel_size=num_conv, strides=(1,1),
                                    padding='same', activation='relu')
_deconv_2_decoded = Conv2DTranspose(filters, kernel_size=num_conv, strides=(1,1),
                                    padding='same', activation='relu')
_x_decoded_relu = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2),
                                  padding='valid', activation='relu')
_x_decoded_mean_squash = Conv2D(img_chns, kernel_size=(2,2), padding='same', activation='sigmoid')

decoder_input = Input(shape=(latent_dim,))
layer1 = _hid_decoded(decoder_input)
layer2 = _up_decoded(layer1)
layer3 = _reshape_decoded(layer2)
layer4 = _deconv_1_decoded(layer3)
layer5 = _deconv_2_decoded(layer4)
layer6 = _x_decoded_relu(layer5)
layer7 = _x_decoded_mean_squash(layer6)
generator = Model(decoder_input, layer7)

_hid_decoded.set_weights(decoder_hid.get_weights())
_up_decoded.set_weights(decoder_upsample.get_weights())
_deconv_1_decoded.set_weights(decoder_deconv_1.get_weights())
_deconv_2_decoded.set_weights(decoder_deconv_2.get_weights())
_x_decoded_relu.set_weights(decoder_deconv_3_upsamp.get_weights())
_x_decoded_mean_squash.set_weights(decoder_mean_squash.get_weights())

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-1, 1] standard deviations
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10,10))
plt.imshow(figure)
plt.savefig('decoder_generator.png')

generator.save(KERAS_MODEL_FILEPATH)

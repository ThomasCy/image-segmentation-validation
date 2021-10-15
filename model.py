from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend
import numpy as np
from config import train_config as config

resized_image_size = config["resized_image_size"]
latent_vector_size = config["latent_vector_size"]
input_tensor_shape = (resized_image_size, resized_image_size, 3)


def conv_block(input_tensor, filter_count, filter_size=3, strides=1, padding='same'):
    input_tensor = layers.Conv2D(
        filter_count, filter_size, strides=strides, padding=padding)(input_tensor)
    input_tensor = layers.ReLU()(input_tensor)
    input_tensor = layers.BatchNormalization()(input_tensor)
    return input_tensor


def t_conv_block(input_tensor, filter_count, filter_size=3, strides=1, padding='same'):
    input_tensor = layers.Conv2DTranspose(
        filter_count, filter_size, strides=strides, padding=padding)(input_tensor)
    input_tensor = layers.ReLU()(input_tensor)
    input_tensor = layers.BatchNormalization()(input_tensor)
    return input_tensor


def build_encoder(input_size, latent_vector_length):
    layer_outputs = []
    input = layers.Input(shape=input_size)
    x = input

    x = conv_block(x, 20)
    layer_outputs.append(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(x, 40)
    layer_outputs.append(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(x, 60)
    layer_outputs.append(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(x, 60)
    layer_outputs.append(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(x, 60)
    layer_outputs.append(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(x, 60)
    layer_outputs.append(x)
    x = layers.MaxPool2D()(x)

    post_conv_tensor_shape = backend.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_vector_length, activation='relu')(x)
    encoder_output = x
    return input, encoder_output, post_conv_tensor_shape, layer_outputs


def build_decoder(latent_vector_length, pre_tconv_shape):
    layer_outputs = []
    decoder_input = layers.Input(latent_vector_length)
    x = decoder_input
    x = layers.Dense(np.prod(pre_tconv_shape[1:]))(x)
    x = layers.Reshape(
        (pre_tconv_shape[1], pre_tconv_shape[2], pre_tconv_shape[3]))(x)
    x = t_conv_block(x, 60, strides=2)
    layer_outputs.append(x)
    x = t_conv_block(x, 60, strides=2)
    layer_outputs.append(x)
    x = t_conv_block(x, 60, strides=2)
    layer_outputs.append(x)
    x = t_conv_block(x, 60, strides=2)
    layer_outputs.append(x)
    x = t_conv_block(x, 40, strides=2)
    layer_outputs.append(x)
    x = t_conv_block(x, 20, strides=2)
    layer_outputs.append(x)
    x = t_conv_block(x, 3, strides=1, padding='same')
    x = layers.Activation('sigmoid')(x)
    decoder_output = x
    return decoder_input, decoder_output,layer_outputs


encoder_input, encoder_output, post_conv_shape,encoder_layer_outputs = build_encoder(
    input_tensor_shape, latent_vector_size)
encoder = Model(encoder_input, encoder_output, name="image_encoder")
encoder.summary()

decoder_input, decoder_output,decoder_layer_outputs = build_decoder(
    latent_vector_size, post_conv_shape)
decoder = Model(decoder_input, decoder_output, name="image_decoder")
decoder.summary()

autoencoder_input = layers.Input(input_tensor_shape)
encoded_image = encoder(autoencoder_input)
decoded_image = decoder(encoded_image)
autoencoder = Model(inputs=autoencoder_input,
                    outputs=decoded_image, name="image_autoencoder")

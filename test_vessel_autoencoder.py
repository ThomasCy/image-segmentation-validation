import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from config import train_config as config
from load_dataset import load_dataset
from model import autoencoder, encoder, decoder, encoder_layer_outputs, encoder_output, decoder_output, decoder_layer_outputs, encoder_input, decoder_input, input_tensor_shape, autoencoder_input

resized_image_size = config["resized_image_size"]
latent_vector_size = config["latent_vector_size"]
input_tensor_shape = (resized_image_size, resized_image_size, 3)

dataset_path = 'uncropped_images/testing_connectedness/'

# Perform reproducible train,dev and test split
x_test, a, b = load_dataset(dataset_path)


def show_sample_images():
    fig = plt.figure(figsize=(100, 10))
    sample_images = 1

    for i in range(1, sample_images+1):
        ax = plt.subplot(1, sample_images, i)
        plt.imshow(x_test[i-1])

    plt.title("Sample images from test set")
    plt.show()


print(f"Number of images in test set:{x_test.shape[0]}")

autoencoder.summary()

autoencoder.load_weights("vessel_conv_autoencoder_final.h5")

encoded_vector = encoder(x_test[:5]).numpy()
torn_imgs, *_ = load_dataset("uncropped_images/Pointy/")
torn_vector = encoder(torn_imgs[:5]).numpy()
torn_similarity_scores = np.dot(
    encoded_vector/np.linalg.norm(encoded_vector), torn_vector.T/np.linalg.norm(torn_vector))
decoded_imgs = autoencoder(x_test[:5]).numpy()


def plot_layer_activations():
    plt.imshow(x_test[0:1][0])
    plt.savefig("input_image.png")
    for i in range(len(decoder_layer_outputs)):
        l1 = K.function([encoder_input], [encoder_output])
        l1_output = l1(x_test[0:1])
        l2 = K.function([decoder_input], [
                        decoder_layer_outputs[i], decoder_output])
        l2_output, decoder_reconstruction = l2(l1_output)
        plt.imshow(decoder_reconstruction[0])
        plt.savefig(f"decoder reconstruction.png")
        plt.imshow(l2_output[0][:, :, 0], cmap="gray")
        plt.savefig(f"layer {i+1} output.png")


def plot_similarity(original_images, torn_images, n_images=5):
    fig = plt.figure(figsize=(20, 4))
    for i in range(1, n_images+1):
        ax = plt.subplot(2, n_images, i)
        plt.imshow(original_images[i-1])
        ax.set_title(
            f"mean similarity:{np.mean(torn_similarity_scores[i-1]):.3f}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for i in range(1, n_images+1):
        ax = plt.subplot(2, n_images, i+n_images)
        plt.imshow(torn_images[i-1])
        # ax.set_title(f"similarity:{torn_similarity_scores[i-1]:.2f}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_reconstruction_images():
    n = 5
    fig = plt.figure(figsize=(5, 2))
    for i in range(n):
        ax = plt.subplot(2, 5, (i + 1) + (i // 5) * 5)
        plt.imshow(x_test[i+5])
        plt.title("orginal")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 5, i + 1 + (i // 5) * 5 + 5)
        plt.imshow(decoded_imgs[i+5])
        plt.title("recontructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

plot_layer_activations()
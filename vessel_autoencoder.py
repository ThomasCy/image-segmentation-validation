import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from PIL import Image
from config import train_config as config
from load_dataset import load_dataset
from model import autoencoder, encoder, decoder

resized_image_size = config["resized_image_size"]
latent_vector_size = config["latent_vector_size"]
dataset_path = 'Resized/combined_flipped/'

# Perform reproducible train, dev and test split
x_train, x_dev, x_test = load_dataset(dataset_path)

# Show a sample of the dataset
fig = plt.figure(figsize=(100, 10))
sample_images = 10

for i in range(1, sample_images+1):
    ax = plt.subplot(1, sample_images, i)
    plt.imshow(x_train[i])

plt.title("Sample images from training set")
plt.show()

print(f"Number of images in training set:{x_train.shape[0]}")
print(f"Number of images in dev set:{x_dev.shape[0]}")
print(f"Number of images in test set:{x_test.shape[0]}")


input_tensor_shape = (resized_image_size, resized_image_size, 3)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# Training starts
autoencoder.fit(x_train, x_train,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                shuffle=True,
                validation_data=(x_dev, x_dev))

encoded_imgs = encoder(x_dev[:20]).numpy()
decoded_imgs = decoder(encoded_imgs).numpy()

n = 5
fig = plt.figure(figsize=(5, 2))
for i in range(n):
    ax = plt.subplot(2, 5, (i + 1) + (i // 5) * 5)
    plt.imshow(x_dev[i])
    plt.title("org.")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 5, i + 1 + (i // 5) * 5 + 5)
    plt.imshow(decoded_imgs[i])
    plt.title("recon.")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# training ends and save the model
autoencoder.save(os.path.join('./', "vessel_conv_autoencoder.h5"))

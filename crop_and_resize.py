import cv2
import numpy as np
import math
import os
from pathlib import Path

for file_name in os.listdir('intra_data/images/good/'):
    if Path(f"intra_data/images/good/{file_name}").is_dir():
        continue

    input_image = cv2.imread(f"intra_data/images/good/{file_name}")
    input_h, input_w, *_ = input_image.shape

    # convert to grayscale
    bw = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(bw, (7, 7), 0)
    bw = 255 - bw

    thresholded_image = cv2.adaptiveThreshold(
        bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # create bounding rectangle to crop the image.
    x, y, w, h = cv2.boundingRect(thresholded_image)

    if h > w:
        x -= min(x, math.floor((h-w)/2))
        w = min(input_w, h)
    else:
        y -= min(y, math.floor((w-h)/2))
        h = min(input_h, w)

    # then crop it and save the images
    crop = input_image[y:y+h, x:x+w]

    resized = cv2.resize(crop, (1024, 1024))
    # flip_1 = cv2.flip(resized, 1)
    # flip_2 = cv2.flip(resized, 0)
    # flip_3 = cv2.flip(resized, -1)
    # rot_1 = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
    # rot_2 = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rot_flip_1 = cv2.rotate(flip_1, cv2.ROTATE_90_CLOCKWISE)
    # rot_flip_2 = cv2.rotate(flip_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"Resized/good/{file_name}_orig.png", resized)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_flip_1.png", flip_1)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_flip_2.png", flip_2)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_flip_3.png", flip_3)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_rot_1.png", rot_1)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_rot_2.png", rot_2)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_rot_flip_1.png", rot_flip_1)
    # cv2.imwrite(f"Resized/combined_flipped/{file_name}_rot_flip_2.png", rot_flip_2)
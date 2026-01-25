import numpy as np


def apply_logistic_map(image, f = 3.6, d_0 = 0.5):

    H, W = image.shape[:2]
    nb_pixels = H * W

    d = np.zeros(nb_pixels)
    d[0] = d_0
    for i in range(1, nb_pixels):
        d[i] = f * d[i-1] * (1 - d[i-1])

    indices = np.argsort(d)

    flat_image = image.reshape(-1, image.shape[2])
    scrambled_flat = flat_image[indices]
    scrambled_image = scrambled_flat.reshape(H, W, image.shape[2])

    return scrambled_image


def invert_logistic_map(scrambled_image, f = 3.6, d_0 = 0.5):

    H, W = scrambled_image.shape[:2]
    nb_pixels = H * W

    d = np.zeros(nb_pixels)
    d[0] = d_0
    for i in range(1, nb_pixels):
        d[i] = f * d[i-1] * (1 - d[i-1])

    indices = np.argsort(np.argsort(d))

    scrambled_flat = scrambled_image.reshape(-1, scrambled_image.shape[2])
    flat_image = scrambled_flat[indices]
    image = flat_image.reshape(H, W, scrambled_image.shape[2])

    return image



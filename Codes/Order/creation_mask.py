import numpy as np


def dummy_mask(image):

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center_x = w // 2
    center_y =  h // 2
    offset_x = w // 4
    offset_y = h // 4

    start_x = center_x - offset_x
    end_x = center_x + offset_x

    start_y = center_y - offset_y
    end_y = center_y + offset_y

    mask[start_y : end_y, start_x : end_x] = 1

    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

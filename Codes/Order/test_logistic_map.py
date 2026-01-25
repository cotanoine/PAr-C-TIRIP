from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import pywt

from logistic_map import *

def open_image(filename:str, mode:str="RGB"):
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    return px



im = open_image(f"Images/test_images/feu.jpg")
mask = np.zeros_like(im)+1

im_shuffle = apply_logistic_map(im,mask)
im2 = reverse_logistic_map(im_shuffle,mask)

plt.subplot(131),plt.imshow(im),plt.title('Image origine')
plt.subplot(132),plt.imshow(im_shuffle),plt.title('Image shuffle')
plt.subplot(133),plt.imshow(im2),plt.title('Image unshuffle')
plt.show()
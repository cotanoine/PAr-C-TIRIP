from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import pywt

from haar_manipulation import *

def open_image(filename:str, mode:str="RGB"):
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    return px



host = open_image(f"Images/test_images/point.jpg")
watermark = open_image(f"Images/test_images/pointW.jpg")
param = 5e-2


res = dwt_embed2(host,watermark,param)
im2 = dwt_extract2(host,res,param)

plt.subplot(221),plt.imshow(host),plt.title('Hote')
plt.subplot(222),plt.imshow(watermark),plt.title('Watermark')
plt.subplot(223),plt.imshow(res),plt.title('embed')
plt.subplot(224),plt.imshow(im2),plt.title('extract')
plt.show()
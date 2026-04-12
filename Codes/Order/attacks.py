
import cv2
from skimage import transform as tf
import numpy as np


def mean_filter(image,size=3):
    return cv2.blur(image,(size, size))

def median_filter(image,size=3):
    return cv2.medianBlur(image, size)

def shear_filter(image):
    afine_tf = tf.AffineTransform(shear=0.4)
    im_shear = tf.warp(image, inverse_map=afine_tf)
    im_shear = np.flipud(im_shear)
    afine_tf = tf.AffineTransform(shear=-0.2)
    im_shear = tf.warp(im_shear, inverse_map=afine_tf)
    return np.flipud(im_shear)
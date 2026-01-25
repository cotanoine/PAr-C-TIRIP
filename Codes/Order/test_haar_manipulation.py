from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from haar_manipulation import *

def open_image(filename:str, mode:str="RGB"):
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    return px



host = open_image(f"Images/test_images/feu.jpg")
watermark = open_image(f"Images/test_images/feuW.jpg")
param = 5e-2


res = apply_dwt(host,watermark,param)
im2 = reverse_dwt(host,res,watermark,param)

plt.subplot(221),plt.imshow(host),plt.title('Hote')
plt.subplot(222),plt.imshow(watermark),plt.title('Watermark')
plt.subplot(223),plt.imshow(res),plt.title('embed')
plt.subplot(224),plt.imshow(im2),plt.title('extract')
plt.show()

plt.imshow(np.abs(watermark-im2))
plt.show()
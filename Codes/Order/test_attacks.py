from image_processing import *
from attacks import *



im = open_image(f"Images/test_images/feuW.jpg")

im_mean = mean_filter(im)
im_median = median_filter(im)
im_shear = shear_filter(im)




# ----------------------------------

import matplotlib.pyplot as plt

plt.subplot(141),plt.imshow(im),plt.title('Image origine')
plt.subplot(142),plt.imshow(im_mean),plt.title('Image Mean')
plt.subplot(143),plt.imshow(im_median),plt.title('Image Median')
plt.subplot(144),plt.imshow(im_shear),plt.title('Image Shear')
plt.show()
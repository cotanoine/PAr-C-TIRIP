
from image_processing import *
from arnold_cat_map import *



im = open_image(f"Images/test_images/feu.jpg")

im_catted = apply_arnold_map(im)
im_recover = reverse_arnold_map(im_catted)




# ----------------------------------

aSSIM(im,im_recover)

import matplotlib.pyplot as plt

plt.subplot(131),plt.imshow(im),plt.title('Image origine')
plt.subplot(132),plt.imshow(im_catted),plt.title('Image Arnold')
plt.subplot(133),plt.imshow(im_recover),plt.title('Image de-Arnold')
plt.show()



from image_processing import *
from logistic_map import *



im = open_image(f"Images/test_images/feu.jpg")
mask = np.zeros_like(im)+1


im_shuffle = apply_logistic_map(im,mask)
im_recover = reverse_logistic_map(im_shuffle,mask)


aSSIM(im,im_recover)


import matplotlib.pyplot as plt

plt.subplot(131),plt.imshow(im),plt.title('Image origine')
plt.subplot(132),plt.imshow(im_shuffle),plt.title('Image shuffle')
plt.subplot(133),plt.imshow(im_recover),plt.title('Image unshuffle')
plt.show()


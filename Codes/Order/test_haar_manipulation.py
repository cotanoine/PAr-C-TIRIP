
from image_processing import *
from haar_manipulation import *




host = open_image(f"Images/test_images/feu.jpg")
watermark = open_image(f"Images/test_images/feuW.jpg")
param = 5e-2


image_watermarked = apply_dwt(host,watermark,param)
watermark_extracted = reverse_dwt(host,image_watermarked,host,param)




# ----------------------------------

aSSIM(watermark,watermark_extracted)

import matplotlib.pyplot as plt

plt.subplot(221),plt.imshow(host),plt.title('Hote')
plt.subplot(222),plt.imshow(watermark),plt.title('Watermark')
plt.subplot(223),plt.imshow(image_watermarked),plt.title('embed')
plt.subplot(224),plt.imshow(watermark_extracted),plt.title('extract')
plt.show()


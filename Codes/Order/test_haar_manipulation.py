
from image_processing import *
from haar_manipulation import *

import math

import matplotlib.pyplot as plt

param = 0.02

host = open_image(f"Images/article/55.png")
watermark = open_image(f"Images/article/C.png")

image_watermarked = apply_dwt(host,watermark,param,'haar')


image_watermarked_scaled = scale_to_uint(np.astype(image_watermarked,np.uint8),host)
image_watermarked = np.astype(image_watermarked,np.uint8)

print(watermark.max())
plt.imshow(image_watermarked_scaled)
plt.show()

watermark_extracted_scaled = reverse_dwt(host,image_watermarked_scaled,watermark,param,'haar')
watermark_extracted = reverse_dwt(host,image_watermarked,watermark,param,'haar')





plt.subplot(221),plt.imshow(host),plt.title('Hote')
plt.subplot(222),plt.imshow(watermark),plt.title('Watermark')
plt.subplot(223),plt.imshow(watermark_extracted_scaled),plt.title('Extraction avec remise à l\'échelle')
plt.subplot(224),plt.imshow(watermark_extracted),plt.title('Extraction')
plt.show()


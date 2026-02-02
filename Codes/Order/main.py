from image_processing import *
from creation_mask import *

from arnold_cat_map import *
from logistic_map import *
from haar_manipulation import *

from blind import *







alpha,beta = 0.04,0.02

host = open_image(f"Images/test_images/feu.jpg").astype(np.float16)
watermark = scale_to_uint2(open_image(f"Images/test_images/feuW.jpg").astype(np.uint8),10,240).astype(np.float16)
watermark = cv2.resize(watermark, (host.shape[1],host.shape[0])).astype(np.float16)

mask = dummy_mask(host).astype(np.float16)
mask = open_image(f"Images/test_images/water.jpg").astype(np.float16)


# --------------------



F_I = host * mask
B_I = host * (1-mask)

F_W = watermark * mask
B_W = watermark * (1-mask)


F_W_arnold = apply_arnold_map(F_W)
B_W_arnold = apply_arnold_map(B_W)

F_W_logistic = apply_logistic_map(F_W_arnold,mask)
B_W_logistic = apply_logistic_map(B_W_arnold,1-mask)

F_final = apply_dwt(F_I, F_W_logistic, alpha)
B_final = apply_dwt(B_I, B_W_logistic, beta)

I_prime = fusion(F_final,B_final,mask)

# --------------------

F_Iw = I_prime * mask
B_Iw = I_prime * (1-mask)

I_blind = I_approximated(I_prime,I_prime)

I_blind = host

F_I_recover = I_blind * mask
B_I_recover = I_blind * (1-mask)

F_W_logistic_recover = reverse_dwt(F_I_recover, F_Iw, F_W_logistic, alpha)
B_W_logistic_recover = reverse_dwt(B_I_recover, B_Iw, B_W_logistic, beta)

F_W_arnold_recover = reverse_logistic_map(F_W_logistic_recover,mask)
B_W_arnold_recover = reverse_logistic_map(B_W_logistic_recover,1-mask)

F_W_recover = reverse_arnold_map(F_W_arnold_recover)
B_W_recover = reverse_arnold_map(B_W_arnold_recover)

W_prime = fusion(F_W_recover,B_W_recover,mask)

# --------------------

aSSIM(W_prime.astype(np.uint8),watermark.astype(np.uint8))

import matplotlib.pyplot as plt

plt.subplot(121),plt.imshow(watermark.astype(np.uint8))
plt.subplot(122),plt.imshow(W_prime.astype(np.uint8))
plt.show()

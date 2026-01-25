import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from arnold_cat_map import *
from logistic_map import *
from haar_manipulation import *
from dummy_mask import *



def open_image(filename:str, mode:str="RGB"):
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    return px

alpha,beta = 0.04,0.02

host = open_image(f"Images/test_images/feu.jpg")
watermark = open_image(f"Images/test_images/feuW.jpg")
watermark = cv2.resize(watermark, (host.shape[1],host.shape[0]))

mask = dummy_mask(host)

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

res = F_final * mask + B_final * (1 - mask)
I_prime = np.clip(res, 0, 255).astype(np.uint8)

# --------------------

F_Iw = I_prime * mask
B_Iw = I_prime * (1-mask)

F_I_recover = F_I
B_I_recover = B_I

F_W_logistic_recover = reverse_dwt(F_I_recover, F_Iw, F_W_logistic, alpha)
B_W_logistic_recover = reverse_dwt(B_I_recover, B_Iw, B_W_logistic, beta)

F_W_arnold_recover = reverse_logistic_map(F_W_logistic_recover,mask)
B_W_arnold_recover = reverse_logistic_map(B_W_logistic_recover,1-mask)

F_W_recover = reverse_arnold_map(F_W_arnold_recover)
B_W_recover = reverse_arnold_map(B_W_arnold_recover)

res = F_W_recover * mask + B_W_recover * (1 - mask)
W_prime = np.clip(res, 0, 255).astype(np.uint8)


# --------------------

plt.subplot(121),plt.imshow(watermark)
plt.subplot(122),plt.imshow(W_prime)
plt.show()

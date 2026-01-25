import cv2
import numpy as np
import matplotlib.pyplot as plt

from arnold_cat_map import arnold_map_inv
from logistic_map import invert_logistic_map
from haar_manipulation import dwt_extract
from dummy_mask import dummy_mask
from blind import I_approximated



def watermarking_process(image_watermarked, host, output, alpha=0.04, beta=0.02):
    
    Iw = cv2.imread(image_watermarked).astype(np.float32)

    if host is not None:
        I = cv2.imread(host).astype(np.float32)
    else:
        I = I_approximated(Iw,Iw)

#--------------------- Application du masque ----------------------------------

    mask = dummy_mask(I)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    F_I = I * mask_3d
    B_I = I * (1 - mask_3d)

    F_Iw = Iw * mask_3d
    B_Iw = Iw * (1 - mask_3d)

#--------------------------- Arnold et mélange -----------------------------

    F_1 = dwt_extract(F_I, F_Iw, alpha)
    B_1 = dwt_extract(B_I, B_Iw, beta)

    F_W_arnold = arnold_map_inv(F_1)
    B_W_arnold = arnold_map_inv(B_1)

    F_Iw_logistic = invert_logistic_map(F_W_arnold)
    B_Iw_logistic = invert_logistic_map(B_W_arnold)



    res = F_Iw_logistic * mask_3d + B_Iw_logistic * (1 - mask_3d)

    I_prime = np.clip(res, 0, 255).astype(np.uint8)

    cv2.imwrite(output, I_prime)





if __name__ == "__main__":

    watermarking_process("Images/output_dummy.png", "Images/host.jpg", "Images/output_dummy_end.png")
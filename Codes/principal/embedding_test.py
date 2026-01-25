import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

from arnold_cat_map import arnold_map
from logistic_map import apply_logistic_map
from haar_manipulation import dwt_embed

def dummy_mask(image):

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center_x = w // 2
    center_y =  h // 2
    offset_x = w // 4
    offset_y = h//4

    start_x = center_x - offset_x
    end_x = center_x + offset_x

    start_y = center_y - offset_y
    end_y = center_y + offset_y

    mask[start_y : end_y, start_x : end_x] = 1
         
    return mask




def watermarking_process(host, watermark, output, alpha=0.04, beta=0.02):
    
    I = cv2.imread(host).astype(np.float32)
    W = cv2.imread(watermark).astype(np.float32)


#--------------------- Application du masque ----------------------------------

    mask = dummy_mask(I)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    """
    # Faire ca ca donne pas les bonnes dimensions
    F_I = np.array([I[:,:,0]*mask, I[:,:,1]*mask, I[:,:,2]*mask])
    B_I = np.array([I[:,:,0]*(1-mask), I[:,:,1]*(1-mask), I[:,:,2]*(1-mask)])
    """

    F_I = I * mask_3d
    B_I = I * (1 - mask_3d)

    # On doit resize W sinon ca a aucun sens d'appliquer le masque de I dessus
    # En fait l'article dit tranquillou que les deux "ont" la même taille au début de 3.2
    W = cv2.resize(W, (I.shape[1],I.shape[0]))

    print(f"W a maintenant une taille {W.shape}")

    F_W = W * mask_3d
    B_W = W * (1 - mask_3d)



    plt.subplot(2,3,1)
    plt.title("Host Image")
    plt.imshow(cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2RGB)) # Sinon les couleurs sont inversées (mais c'est marrant)

    plt.subplot(2,3,4)
    plt.title("Watermark Image")
    plt.imshow(cv2.cvtColor(W.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(2,3,2)
    plt.title("Host Image Foreground")
    plt.imshow(cv2.cvtColor(F_I.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(2,3,3)
    plt.title("Host Background")
    plt.imshow(cv2.cvtColor(B_I.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(2,3,5)
    plt.title("Watermark  Foreground")
    plt.imshow(cv2.cvtColor(F_W.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(2,3,6)
    plt.title("Watermark Background")
    plt.imshow(cv2.cvtColor(B_W.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.show()


    print(f"Les images ont la forme {I.shape}")

#--------------------------- Arnold et mélange -----------------------------

    F_W_arnold = arnold_map(F_W)
    B_W_arnold = arnold_map(B_W)


    F_W_logistic = apply_logistic_map(F_W_arnold)
    B_W_logistic = apply_logistic_map(B_W_arnold)


    res = np.zeros_like(I)

    for k in range(3):

        F_res = dwt_embed(F_I[:, :, k], F_W_logistic[:, :, k], alpha)
        
        B_res = dwt_embed(B_I[:, :, k], B_W_logistic[:, :, k], beta)

        res[:, :, k] = (F_res * mask) + (B_res * (1 - mask))


    I_prime = np.clip(res, 0, 255).astype(np.uint8)

    cv2.imwrite(output, I_prime)





if __name__ == "__main__":

    watermarking_process("Images/host.jpg", "Images/watermark.jpg", "Images/output_dummy.png")
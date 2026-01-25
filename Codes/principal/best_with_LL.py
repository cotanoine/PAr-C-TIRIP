import pywt
import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    Mon_image = cv2.imread("Images/watermark.jpg").astype(np.float32)

    res = np.zeros_like(Mon_image)

    for k in range(3):

        LL_k, _ = pywt.dwt2(Mon_image[:,:,k], 'haar')
        zeros = np.zeros_like(LL_k)

        coeffs = (LL_k, (zeros, zeros, zeros))
        from_LL_k = pywt.idwt2(coeffs, 'haar')
        from_LL_k = cv2.resize(from_LL_k, (Mon_image.shape[1], Mon_image.shape[0]))

        res[:,:,k] = from_LL_k
    
    res = np.clip(res, 0, 255).astype(np.uint8)

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(Mon_image.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(1,2,2)
    plt.title("Best Reconstitution from LL")
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

    plt.show()
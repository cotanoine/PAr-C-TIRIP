import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load grayscale image
img = cv2.imread("Images/test_images/lyon.png", cv2.IMREAD_GRAYSCALE)

# Define Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply convolution using SciPy
laplacian_custom = ndimage.convolve(img.astype(np.int32), laplacian_kernel)

# Normalize for display
laplacian_custom = np.clip(laplacian_custom, 0, 255).astype(np.uint8)

# Plot results
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(laplacian_custom, cmap='gray'), plt.title('Custom Laplacian Filter')
plt.show()

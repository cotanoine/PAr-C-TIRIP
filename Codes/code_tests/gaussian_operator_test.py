import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import feature, color

# Load image with Pillow
img = Image.open("Images/test_images/lyon.png").convert("L")
img_array = np.array(img)

sigma = 1

coarse = gaussian_filter(img_array, sigma=sigma)

# Apply Canny edge detector (scikit-image)
edges = feature.canny(img_array, sigma=sigma)

Ec_ = 255 * (np.ones(shape=coarse.shape) - np.uint32(edges))
Ec = np.clip(np.uint32(coarse) + Ec_,0,255).astype(np.uint8)
#Ec = np.clip( 255 - (np.uint32(coarse)  + 255*(np.uint32(edges) - np.ones(shape=coarse.shape))),0,255).astype(np.uint8)


Ec_test = np.clip( (np.uint32(edges) - np.ones(shape=coarse.shape)),-1,255)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(coarse, cmap="gray")
axes[1].set_title(f"Coarse Image (σ={sigma})")
axes[1].axis("off")

axes[2].imshow(edges, cmap="gray")
axes[2].set_title(f"Canny Edges (σ={sigma})")
axes[2].axis("off")

axes[3].imshow(Ec, cmap="gray")
axes[3].set_title("Enhanced Coarse Image")
axes[3].axis("off")



plt.show()

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt

def open_image(filename:str, mode:str="RGB"):
    
    im = Image.open(filename).convert(mode=mode)
    px = im.load()
    
    return im, px
       

def gaussian_gradient_operator_convolutor(A,size=5, sigma=1.0):
    """
    Returns Gaussian gradient operator kernels Gx, Gy.
    size: odd number (e.g., 3, 5, 7)
    sigma: standard deviation of the Gaussian
    """
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))

    # Gaussian (no normalization needed for derivative operator)
    G = np.exp(-(x*x + y*y) / (2*sigma*sigma))

    # Gaussian derivatives
    Gx = -(x / (sigma*sigma)) * G
    Gy = -(y / (sigma*sigma)) * G

    return np.sqrt(convolve2d(A, Gx,'same')**2 + convolve2d(A,Gy,'same')**2)

def to_uint8(x):
    return np.clip(x,0,255).astype(np.uint8)


#-------------------------------------------------------------------------------
#                               Step 1
#-------------------------------------------------------------------------------

def step_one(I_filename, W_filename, display=False):

    # I : the host image, size (height_pixels, width_pixels)
    # W : the watermark image (same size as I)

    I, I_pixels = open_image(I_filename)
    W, W_pixels = open_image(W_filename)
    #if display:
        #I.show()
        #W.show()


    A, A_pixels = open_image(I_filename, mode="L")
    #A.show()

    # A : The grayscale image of I
    A_array = np.array(A)

    sigma = np.std(A_array)
    print(f"On a un écart-type {sigma}")
    #print(A_array[400])

    sigma = 2

    # The coarse image, convolution of the gaussian gradient operator and A.
    #g = gaussian_gradient_operator_convolutor(A_array,7,sigma=sigma)
    g = gaussian_filter(A_array,sigma)
    g = np.array(g, dtype=np.uint8)

    g_im = Image.fromarray(g.clip(min=0, max=255).astype(np.uint8))
    #g_im.show()
    


    # valeurs de seuil à determiner
    epsilon = cv2.Canny(image=g.astype(np.uint8), threshold1=100, threshold2 = 200)
    epsilon = np.array(epsilon, dtype=np.uint8)

    edges_img = Image.fromarray(epsilon.astype(np.uint8))
    #edges_img.show()


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(g, cmap='gray')
    axes[0].set_title("g, the coarse image")


    axes[1].imshow(epsilon, cmap='gray')
    axes[1].set_title("epsilon, the coarse edges")


    #Ec_unsaturated = np.int32(g) + 255 - np.int32(epsilon)
    #Ec = np.clip(Ec_unsaturated,0,255).astype(np.uint8)

    #buffer = 255 - epsilon
    Ec = np.clip(g + 255 - 255*epsilon,0,255).astype(np.uint8)


    axes[2].imshow(Ec, cmap='gray')
    axes[2].set_title("Ec, the enhanced coarse image")

    plt.show()

    print(np.max(g))
    print(np.min(g))

    print(np.max(epsilon))
    print(np.min(epsilon))

    print(np.max(Ec))
    print(np.min(Ec))
    Ec_im = Image.fromarray(Ec)#.astype(np.uint8))
    #Ec_im.show()

    #if display:
        #coarsed_enhanced_img = Image.fromarray(Ec)
        #coarsed_enhanced_img.show()


    laplacian_kernel = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]])

    # Apply convolution using SciPy
    laplacian_custom = convolve(A_array.astype(np.int32), laplacian_kernel)

    # Normalize for display
    laplacian = np.clip(laplacian_custom, 0, 255).astype(np.uint8)
    #laplacian = np.uint8(np.absolute(laplacian))

    plt.subplot(221), plt.imshow(A_array, cmap='gray'), plt.title('Original')
    plt.subplot(222), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Filter')
    somme_laplacienne = to_uint8(np.int32(A_array) + np.int32(laplacian))
    plt.subplot(223), plt.imshow(somme_laplacienne, cmap='gray'), plt.title('Somme')
    plt.show()

# G_sigma : the gradient Gaussian operator
"""What is sigma the std of ?"""

# g : The coarse image, a convolution of G_sigma and A

# epsilon : coarse edges, by applying the Canny edge detector on g

# Ec = g + 255 - epsilon, the coarse edge-enhanced image

# Ef = A + convolution(A, L), with L the Laplacian filter, the fine edge-enhanced image

# E = Ec+ Ef, the accumulated edges


My_image = "Images/test_images/lyon.png"
My_watermark = "Images/test_images/Untitled.jpeg"

step_one(My_image, My_watermark, display=True)
#-------------------------------------------------------------------------------
#                               Step 2
#-------------------------------------------------------------------------------

# Ed : morphological dilation on E with structuring Element [1,0,1;0,1,0;1,0,1]

# Ee : Histogram equalization on Ed
# n : vector of the regions in Ee, named patches. Let's name its size nb_patches



#-------------------------------------------------------------------------------
#                               Step 3
#-------------------------------------------------------------------------------

# m : size (nb_patches, height_pixels), such that m[j][x] represents the set of pixels of the j-th patch on the x-th row







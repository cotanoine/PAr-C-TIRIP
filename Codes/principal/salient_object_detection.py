

#-------------------------------------------------------------------------------
#                               Step 1
#-------------------------------------------------------------------------------

# I : the host image, size (height_pixels, width_pixels)
# W : the watermark image (same size as I)

# A : The grayscale image of I

# G_sigma : the gradient Gaussian operator
"""What is sigma the std of ?"""

# g : The coarse image, a convolution of G_sigma and A

# epsilon : coarse edges, by applying the Canny edge detector on g

# Ec = g + 255 - epsilon, the coarse edge-enhaced image

# Ef = A + convolution(A, L), with L the Laplacian filter, the fine edge-enhanced image

# E = Ec+ Ef, the accumulated edges



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







""" Using E - SMD"""

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


"""
Ok NO, basically you have to update the arrays m[j][x] with values of a gaussian sum
but the document doesn't make any sense in how you update each cell
"""

#-------------------------------------------------------------------------------
#                               Step 4
#-------------------------------------------------------------------------------

# M : the image represented by the union of the patches (the m[j])

# Apply H1 =  [1, 1, 0, 1, 1;        then H2 = [1, 0, 1;
#              1, 0, 0, 0, 1;                   0, 0, 0;
#              0, 0, 0, 0, 0;                   1, 0, 1]
#              1, 0, 0, 0, 1;
#              1, 1, 0, 1, 1]
#
# M_prime : the result of convolution(convolution(M,H1),H2)

# Update the (values) of patches m[j] as the value of M_prime at spot m[j]

# While there are more patches than the predefined threshold tau_nu 
        # theta : such that m[theta] has the least pixels of all the m[j]

        # Identify the adjacent patches

        # Merge m[theta] with the largest adjacent patch m[lambda_theta]:
"""     Do we define m as a dictionnary of arrays for simplicity ? """

# M : output of this step
""" Should have at most tau_nu colors"""


#-------------------------------------------------------------------------------
#                               Step 5
#-------------------------------------------------------------------------------

# M = 255 - M

# Apply H3 = 1/13 * [1, 1, 0, 1, 1;         then H4 = 1/10 *  [2, 0, 2; 
#                    1, 0, 0, 0, 1;                            0, 2, 0;
#                    0, 0, 1, 0, 0;                            2, 0, 2]
#                    1, 0, 0, 0, 1;
#                    1, 1, 0, 1, 1]
#
# S: Saliency map = convolution(convolution(M,H3),H4)

# Sigma : the binary segmentation mask, ie, the pixels 
#         where S is higher than the predefined threshold tau_s



import numpy as np


def logisticmap(f, d_0, length):
    d = [f*d_0*(1-d_0)]
    for _ in range(length-1):
       d.append(f*d[-1]*(1-d[-1]))
    return np.array(d)

f = 3.92
d_0 = 0.2


def apply_logistic_map(image, mask, f = f, d_0 = d_0):

    H, W = image.shape[:2]

    nz = 0
    for i in range(H):
        for j in range(W):
            if mask[i,j,0] == 0:
                nz += 1

    print(nz)

    d = logisticmap(f, d_0, H*W-nz)
    p = np.argsort(d)
    # p contient les permutations pour re-ordonner dans le "désordre"
    

    image_flat = np.zeros((1,H*W-nz,3))
    indice = 0
    for i in range(H):
        for j in range(W):
            if not mask[i,j,0] == 0:
                image_flat[0,indice,:] = image[i,j]
                indice += 1
    # On aplatit notre matrice n*m en une ligne 1*mn

    image_flat = image_flat[:,p,:]
    # On ordonne les termes sur notre ligne
    
    res = np.zeros_like(image)

    indice = 0
    for i in range(H):
        for j in range(W):
            if not mask[i,j,0] == 0:
                res[i,j,:] = image_flat[0,indice,:]
                indice += 1   
    # On reconstruit la matrice n*m

    return res


def reverse_logistic_map(scrambled_image, mask, f = f, d_0 = d_0):

    H, W = scrambled_image.shape[:2]

    nz = 0
    for i in range(H):
        for j in range(W):
            if (mask[i,j] == [0,0,0]).all():
                nz += 1


    d = logisticmap(f, d_0, H*W-nz)
    p = np.argsort(np.argsort(d))
    # p contient les permutations pour re-ordonner dans le "désordre"


    scrambled_image_flat = np.zeros((1,H*W-nz,3))
    indice = 0
    for i in range(H):
        for j in range(W):
            if not mask[i,j,0] == 0:
                scrambled_image_flat[0,indice,:] = scrambled_image[i,j]
                indice += 1
    # On aplatit notre matrice n*m en une ligne 1*mn


    scrambled_image_flat = scrambled_image_flat[:,p,:]
    # On ordonne les termes sur notre ligne
    
    res = np.zeros_like(scrambled_image)

    indice = 0
    for i in range(H):
        for j in range(W):
            if not mask[i,j,0] == 0:
                res[i,j,:] = scrambled_image_flat[0,indice,:]
                indice += 1   
    # On reconstruit la matrice n*m

    return res




import numpy as np
from scipy.signal import convolve2d

def I_approximated(I,B):
    omega = [0.4,0.35,0.25]
    H4 = np.ones((3,3))/9

    rgb = []

    for k in range(3):
        Ik = I[:,:,k]*1
        Bk = B[:,:,k]
        maxk = Ik.max()
        muk = Bk.mean()

        Ik = Ik - omega[k]*(maxk-muk)/2 # equation (12)

        # Conversion des valeurs de la matrice en pixel (0-255)
        max,min = Ik.max(),Ik.min()
        Ik = 255*(Ik-min)/(max-min)
        
        rgb.append(convolve2d(Ik,H4,'same'))

    return np.dstack((rgb[0],rgb[1],rgb[2])).astype(np.uint8)
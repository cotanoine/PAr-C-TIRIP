import numpy as np
import random

M = 255

def apply_arnold_map(image, M = M, c = (1,1,1,1)):


    H, W = image.shape[:2]
    res = np.zeros_like(image)

    Gamma = np.array([  [1,       c[0],                  c[1]],
                        [c[2],    1+c[0]*c[2],           c[1]*c[2]],
                        [c[3],    c[0]*c[1]*c[2]*c[3],   1+c[1]*c[3]]
                    ])

    #print(f"On a la valeur du premier pixel: {image[0,0]}")
    #print(f"Ca donne {Gamma @ image[0,0]}")

    for i in range(H):
        for j in range(W):
            res[i,j] = (Gamma @ image[i,j] ) % M

    return res


def reverse_arnold_map(image, M = M, c = (1,1,1,1)):

    H, W = image.shape[:2]
    res = np.zeros_like(image)

    Gamma = np.array([  [1,       c[0],                  c[1]],
                        [c[2],    1+c[0]*c[2],           c[1]*c[2]],
                        [c[3],    c[0]*c[1]*c[2]*c[3],   1+c[1]*c[3]]
                    ])
    
    Gamma_inv = np.linalg.inv(Gamma)
    
    #print(f"On a la valeur du premier pixel: {image[0,0]}")
    #print(f"Ca donne {Gamma @ image[0,0]}")

    for i in range(H):
        for j in range(W):
            res[i,j] = (Gamma_inv @ image[i,j] ) % M

    return res
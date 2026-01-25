import numpy as np


def arnold_map(image, M = 512, c = (1,2,3,4)):

    H, W = image.shape[:2]

    Gamma = np.array([  [1,       c[0],                  c[1]],
                        [c[2],    1+c[0]*c[2],           c[1]*c[2]],
                        [c[3],    c[0]*c[1]*c[2]*c[3],   1+c[1]*c[3]]
                    ])

    print(f"On a la valeur du premier pixel: {image[0,0]}")
    print(f"Ca donne {Gamma @ image[0,0]}")

    for i in range(H):
        for j in range(W):
            image[i,j] = (Gamma @ image[i,j] )% M

    return image
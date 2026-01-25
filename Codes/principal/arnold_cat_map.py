import numpy as np
import random

def arnold_map(image, M = 256, c = (1,1,1,1)):

    H, W = image.shape[:2]

    Gamma = np.array([  [1,       c[0],                  c[1]],
                        [c[2],    1+c[0]*c[2],           c[1]*c[2]],
                        [c[3],    c[0]*c[1]*c[2]*c[3],   1+c[1]*c[3]]
                    ])

    #print(f"On a la valeur du premier pixel: {image[0,0]}")
    #print(f"Ca donne {Gamma @ image[0,0]}")

    for i in range(H):
        for j in range(W):
            image[i,j] = (Gamma @ image[i,j] )#% M

    return image


def arnold_map_inv(image, M = 256, c = (1,1,1,1)):

    H, W = image.shape[:2]

    Gamma = np.array([  [1,       c[0],                  c[1]],
                        [c[2],    1+c[0]*c[2],           c[1]*c[2]],
                        [c[3],    c[0]*c[1]*c[2]*c[3],   1+c[1]*c[3]]
                    ])
    
    Gamma_inv = np.linalg.inv(Gamma)
    
    #print(f"On a la valeur du premier pixel: {image[0,0]}")
    #print(f"Ca donne {Gamma @ image[0,0]}")

    for i in range(H):
        for j in range(W):
            image[i,j] = (Gamma_inv @ image[i,j] )#% M

    return image



if __name__ == "__main__":


    Mon_Image = np.zeros((1,1,3))
    Mon_Pixel = random.sample(range(256), 3)
    Mon_Image[0,0] = Mon_Pixel

    print(f"Le pixel original est {Mon_Pixel}")

    Image_melange = arnold_map(Mon_Image)

    print(f"Le pixel mélangé vaut {Image_melange[0,0]}")

    Image_retrouve = arnold_map_inv(Image_melange)

    print(f"On retrouve le pixel : {Image_retrouve[0,0]}")
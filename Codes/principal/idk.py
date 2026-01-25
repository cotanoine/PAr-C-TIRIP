from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import pywt


'''
Ouvre une image et la renvoie sous la forme d'un tableau numpy
'''
def open_image(filename:str, mode:str="RGB"):
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    return px
       

'''
equation (9) et (10)
'''
def truc(App,A,const):

    rgb = []

    for k in range(3):

        rpp = App[:,:,k]
        LLpp,coeffs = pywt.dwt2(rpp, 'haar')
        r = A[:,:,k]
        LL,coeffs = pywt.dwt2(r, 'haar')

        LLe = (LLpp-LL)/const

        rgb.append(pywt.idwt2((LLe,coeffs), 'haar'))


    A_w = np.dstack((rgb[0],rgb[1],rgb[2])).astype(np.uint8)

    return A_w
       
'''
Suite logique de la forme x(i+1) = x(i)[1-x(i)]*r, avec length éléments.
'''
def logisticmap(x_init, r, length):
    x = [r*x_init*(1-x_init)]
    for _ in range(length-1):
       x.append(r*x[-1]*(1-x[-1]))
    return np.array(x)

'''
Re-ordonne les pixels non nuls de la matrice avec la suite logique
'''
def unshuffle(x_init,r,A):
    n = len(A)
    m = len(A[0])

    nz = 0
    for i in range(n):
        for j in range(m):
            if (A[i,j] == [0,0,0]).all():
                nz += 1


    x = logisticmap(x_init, r, n*m-nz)
    p = np.argsort(np.argsort(x))
    # p contient les permutations pour re-ordonner dans le "désordre"


    A_flat = np.zeros((1,n*m-nz,3)).astype(np.uint8)
    indice = 0
    for i in range(n):
        for j in range(m):
            if not (A[i,j] == [0,0,0]).all():
                A_flat[0,indice,:] = A[i,j]
                indice += 1
    # On aplatit notre matrice n*m en une ligne 1*mn


    A_flat = A_flat[:,p,:]
    # On ordonne les termes sur notre ligne
    

    indice = 0
    for i in range(n):
        for j in range(m):
            if not (A[i,j] == [0,0,0]).all():
                A[i,j,:] = A_flat[0,indice,:]
                indice += 1   
    # On reconstruit la matrice n*m



'''
Multiplie la matrice A avec la matrice d'Arnold M
'''
def Arnold_map(A,M):
    n = len(A)
    m = len(A[0])
    
    for i in range(n):
        for j in range(m):
            A[i,j] = (M @ A[i,j,:])%255


'''
Extraction du watermark avec Fw,Bw et le mask : equation (11)
'''
def watermark(F_w,B_w,Mask):
    n = len(Mask)
    m = len(Mask[0])

    W = np.zeros((n,m,3)).astype(np.uint8)

    for i in range(n):
        for j in range(m):
            if Mask[i,j,0] == 1:
                W[i,j] = F_w[i,j]
            else:
                W[i,j] = B_w[i,j]

    return W


'''
Calcul l'approxiamtion de l'image hôte
'''
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








if __name__ == "__main__":

    # Données pour l'extraction ---------------------------
    alpha = 0.04
    beta = 0.02

    x0 = 0.2
    r = 3.92

    c1,c2,c3,c4 = 1,1,1,1
    Arnold = np.array([[1,c1,c2],[c3,1+c1*c3,c2*c3],[c4,c1*c2*c3*c4,1+c2*c4]])
    M = 512
    # -----------------------------------------------------


    #tatou,mask = "lyon","B"
    tatou,mask = "ex","exB"

    # Notre image tatouée
    Watermarked_Image = open_image(f"Images/test_images/{tatou}.png")
    # ---
    
    # Approximation de l'image hôte
    I_approx = I_approximated(Watermarked_Image,Watermarked_Image) # c'est pas B ?
    # ---

    # Déteermination du masque avec l'image hôte
    Mask = open_image(f"Images/test_images/{mask}.png")//255
    # ---

    # Création des images "Foreground" et "Background" pour l'image tatouée
    F_i = Mask * I_approx
    B_i = (1-Mask) * I_approx
    # ---
    
    # Création des images "Foreground" et "Background" pour l'image tatouée
    Fpp_i = Mask * Watermarked_Image
    Bpp_i = (1-Mask) * Watermarked_Image
    # ---

    # Calcul de F_w
    F_w = truc(Fpp_i,F_i,alpha)
    unshuffle(x0,r,F_w)
    Arnold_map(F_w,np.invert(Arnold))
    # ---

    # Calcul de B_w
    B_w = truc(Bpp_i,B_i,beta)
    unshuffle(x0,r,B_w)
    Arnold_map(B_w,np.invert(Arnold))
    # ---

    # Extraction du tatouage
    W = watermark(F_w,B_w,Mask)
    # ---

    # Plots
    plt.subplot(231),plt.imshow(Watermarked_Image),plt.title('Image "tatouée"')
    plt.subplot(232),plt.imshow(I_approx),plt.title('"Image hôte"')
    plt.subplot(233),plt.imshow(W),plt.title('Tatouage')
    plt.subplot(234),plt.imshow(255*Mask),plt.title('Masque')
    plt.subplot(235),plt.imshow(F_w),plt.title('Watermark Foreground')
    plt.subplot(236),plt.imshow(B_w),plt.title('Watermark Background')
    plt.show()








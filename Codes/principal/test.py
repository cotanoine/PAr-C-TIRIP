from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import pywt

def open_image(filename:str, mode:str="RGB"):
    
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    
    return px

def logisticmap(x_init, r, length):
    x = [r*x_init*(1-x_init)]
    for t in range(length):
       x.append(r*x[-1]*(1-x[-1]))
    return np.array(x)



x0 = 0.2
r = 3.92


I = open_image("Images/test_images/ex.png")

N = len(I)
M = len(I[0])


a = np.random.randint(0,10,10)
aa = np.argsort(a)
aaa = np.argsort(aa)
print(a) # original
print(a[aa]) # sorted
print(a[aa][aaa]) # undone

x = logisticmap(0.2, 3.92, N*M-1)
p = np.argsort(x)




If = np.zeros((1,N*M,3)).astype(np.uint8)
for i in range(N):
    for j in range(M):
        If[0,N*i+j,:] = I[i,j]






If = If[:,p,:]

Is = np.zeros((N,M,3)).astype(np.uint8)

for i in range(N):
    for j in range(M):
        Is[i,j,:] = If[0,N*i+j,:]



def unshuffle(x_init,r,A):
    N = len(A)
    M = len(A[0])

    x = logisticmap(0.2, 3.92, N*M-1)
    p = np.argsort(np.argsort(x))

    A_flat = np.zeros((1,N*M,3)).astype(np.uint8)

    for i in range(N):
        for j in range(M):
            A_flat[0,N*i+j,:] = A[i,j]

    A_flat = A_flat[:,p,:]

    A_unshuffle = np.zeros((N,M,3)).astype(np.uint8)

    for i in range(N):
        for j in range(M):
            A_unshuffle[i,j,:] = A_flat[0,N*i+j,:]

    return A_unshuffle


plt.imshow(unshuffle(x0,r,Is))
plt.show()


F = np.array([[1,2,3],[4,5,6]])
print(F.max())
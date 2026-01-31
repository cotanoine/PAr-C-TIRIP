import numpy as np
from PIL import Image

from SSIM_PIL import compare_ssim



def open_image(filename:str, mode:str="RGB"):
    im = Image.open(filename).convert(mode=mode)
    px = np.asarray(im)
    return px


def fusion(foreground,background,mask):
    res = foreground * mask + background * (1 - mask)
    return np.clip(res, 0, 255).astype(np.uint8)


def aSSIM(img1,img2):
    value = compare_ssim(Image.fromarray(img1, 'RGB'), Image.fromarray(img2, 'RGB'))
    print(f"\033[32m~~~ aSSIM = {100*value:.2f}% ~~~\033[0m")



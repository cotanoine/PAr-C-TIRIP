import pywt
import numpy as np
import cv2



def scale_to_uint(image,image_ref):
    mx,mn = image.max(),image.min()
    b_sup,b_inf = image_ref.max(),image_ref.min()
    return (image-mn)*(b_sup-b_inf)/(mx-mn) + b_inf

def scale_to_uint2(image,b_inf,b_sup):
    mx,mn = image.max(),image.min()
    b_inf = max(b_inf,mn)
    b_sup = min(b_sup,mx)
    return ((image.astype(np.float32)-mn)*(b_sup-b_inf)/(mx-mn) + b_inf).astype(np.uint8)

    
wavelet = "haar"



def apply_dwt(host, watermark, param, wavelet):


    res = np.zeros_like(host).astype(np.float32)

    for k in range(3):

        coeffs_h = pywt.dwt2(host[:, :, k], wavelet=wavelet)
        LL_h, (LH_h, HL_h, HH_h) = coeffs_h

        coeffs_w = pywt.dwt2(watermark[:, :, k], wavelet=wavelet)
        LL_w, _ = coeffs_w

        LL_new = LL_h + (param * LL_w)

        # 4. Reconstruction IDWT
        coeffs_new = (LL_new, (LH_h, HL_h, HH_h))

        res_k = pywt.idwt2(coeffs_new, wavelet=wavelet)
        res_k = cv2.resize(res_k, (host.shape[1], host.shape[0]))

        res[:, :, k] = res_k

    return res

def reverse_dwt(host, watermarked, watermark, param, wavelet):

    res = np.zeros_like(host)

    for k in range(3):

        LL_original, _ = pywt.dwt2(host[:, :, k], wavelet=wavelet)
        LL_watermarked, _ = pywt.dwt2(watermarked[:, :, k], wavelet=wavelet)
        _ , truc = pywt.dwt2(watermark[:, :, k], wavelet=wavelet)

        LL_extracted = (LL_watermarked - LL_original) / param

        zeros = np.zeros_like(LL_extracted)
        coeffs_extracted = (LL_extracted, truc)

        res_k = pywt.idwt2(coeffs_extracted, wavelet=wavelet)
        res_k = cv2.resize(res_k, (host.shape[1], host.shape[0]))

        res[:, :, k] = res_k

    return res

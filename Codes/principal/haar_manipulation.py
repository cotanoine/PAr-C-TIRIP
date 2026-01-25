import pywt
import numpy as np
import cv2



def scale_to_uint(image):
    mx,mn = image.max(),image.min()
    b_sup,b_inf = min(255,mx),max(0,mn)
    return (image-mn)*(b_sup-b_inf)/(mx-mn) + b_inf

def scale_to_uint2(image,b_inf,b_sup):
    mx,mn = image.max(),image.min()
    b_inf = max(b_inf,mn)
    b_sup = min(b_sup,mx)
    return ((image.astype(np.float32)-mn)*(b_sup-b_inf)/(mx-mn) + b_inf).astype(np.uint8)

    




def dwt_embed(host, watermark, param):


    res = np.zeros_like(host)

    for k in range(3):

        coeffs_h = pywt.dwt2(host[:, :, k], 'haar')
        LL_h, (LH_h, HL_h, HH_h) = coeffs_h

        coeffs_w = pywt.dwt2(watermark[:, :, k], 'haar')
        LL_w, _ = coeffs_w

        LL_new = LL_h + (param * LL_w)

        # 4. Reconstruction IDWT
        coeffs_new = (LL_new, (LH_h, HL_h, HH_h))

        res_k = np.clip((pywt.idwt2(coeffs_new, 'haar').astype(np.float32)),0,255)
        res_k = cv2.resize(res_k, (host.shape[1], host.shape[0]))

        res[:, :, k] = res_k.astype(np.uint8)

    return res

def dwt_extract(host, watermarked, param):

    res = np.zeros_like(host)

    for k in range(3):

        LL_original, _ = pywt.dwt2(host[:, :, k], 'haar')
        LL_watermarked, _ = pywt.dwt2(watermarked[:, :, k], 'haar')

        LL_extracted = (LL_watermarked - LL_original) / param

        zeros = np.zeros_like(LL_extracted)
        coeffs_extracted = (LL_extracted, (zeros, zeros, zeros))

        res_k = np.clip((pywt.idwt2(coeffs_extracted, 'haar').astype(np.float32)),0,255)
        res_k = cv2.resize(res_k, (host.shape[1], host.shape[0]))

        res[:, :, k] = res_k.astype(np.uint8)

    return res



def dwt_embed2(host, watermark, param, scale):

    host_scale = scale_to_uint2(host,scale[0],scale[1])

    res = np.zeros_like(host)

    for k in range(3):

        coeffs_h = pywt.dwt2(host_scale[:, :, k], 'haar')
        LL_h, (LH_h, HL_h, HH_h) = coeffs_h

        coeffs_w = pywt.dwt2(watermark[:, :, k], 'haar')
        LL_w, _ = coeffs_w

        LL_new = LL_h + (param * LL_w)

        # 4. Reconstruction IDWT
        coeffs_new = (LL_new, (LH_h, HL_h, HH_h))

        res_k = pywt.idwt2(coeffs_new, 'haar')
        res_k = cv2.resize(res_k, (host.shape[1], host.shape[0]))

        res[:, :, k] = res_k

    return res

def dwt_extract2(host, watermarked, param, scale):

    host_scale = scale_to_uint2(host,scale[0],scale[1])

    res = np.zeros_like(host)

    for k in range(3):

        LL_original, _ = pywt.dwt2(host_scale[:, :, k], 'haar')
        LL_watermarked, _ = pywt.dwt2(watermarked[:, :, k], 'haar')

        LL_extracted = (LL_watermarked - LL_original) / param

        zeros = np.zeros_like(LL_extracted)
        coeffs_extracted = (LL_extracted, (zeros, zeros, zeros))

        res_k = pywt.idwt2(coeffs_extracted, 'haar')
        res_k = cv2.resize(res_k, (host.shape[1], host.shape[0]))

        res[:, :, k] = res_k

    return res
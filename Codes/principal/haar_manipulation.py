import pywt
import numpy as np


def dwt_embed(host, watermark, param):

    coeffs_h = pywt.dwt2(host, 'haar')
    LL_h, (LH_h, HL_h, HH_h) = coeffs_h

    coeffs_w = pywt.dwt2(watermark, 'haar')
    LL_w, _ = coeffs_w

    LL_new = LL_h + (param * LL_w)

    # 4. Reconstruction IDWT
    coeffs_new = (LL_new, (LH_h, HL_h, HH_h))
    res = pywt.idwt2(coeffs_new, 'haar')

    return res

def dwt_extract(original, watermarked, param):

    LL_original, _ = pywt.dwt2(original, 'haar')
    LL_watermarked, _ = pywt.dwt2(watermarked, 'haar')

    LL_extracted = (LL_watermarked - LL_original) / param

    zeros = np.zeros_like(LL_extracted)
    coeffs_extracted = (LL_extracted, (zeros, zeros, zeros))
    res = pywt.idwt2(coeffs_extracted, 'haar')

    return res
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

img = cv2.imread("img3.png", cv2.IMREAD_GRAYSCALE)

blocks_sharpness = []
blocks_16x16 = []
blocks_8x8 = []

windowsize_r = 16
windowsize_c = 16

alpha = 1

reversed_row_size = int(img.shape[0] / windowsize_r)
reversed_col_size = int(img.shape[1] / windowsize_c)
img = img[: reversed_row_size * windowsize_r][: reversed_col_size * windowsize_c]
img = img.astype(np.float64)

for r in range(0, img.shape[0], windowsize_r):
    for c in range(0, img.shape[1], windowsize_c):
        block = img[r:r+windowsize_r, c:c+windowsize_c]
        blocks_16x16.append(block)
        blocks_sharpness.append(cv2.Laplacian(block, cv2.CV_64F).var())
    
blocks_with_30percent_highest_sharpness = np.array(blocks_sharpness)      
count_of_30percent = int(.3 * len(blocks_sharpness))  
top_30percent_highest_sharpness_indices = np.sort(np.argpartition(blocks_with_30percent_highest_sharpness, -count_of_30percent)[-count_of_30percent:])
  
for idx in top_30percent_highest_sharpness_indices:
    big_block = blocks_16x16[idx]
    windowsize_r = 8
    windowsize_c = 8
    small_blocks = []
    for r in range(0, big_block.shape[0], windowsize_r):
        for c in range(0, big_block.shape[1], windowsize_c):
            small_blocks.append(big_block[r:r+windowsize_r, c:c+windowsize_c])
    blocks_8x8.append(small_blocks)
    
def dct(block, big=1):
    if big:
        dct = fft.dct(block)
        return dct
    else:
        block_dct = []
        for smaller_block in block:
            dct = fft.dct(smaller_block)
            block_dct.append(dct)
        return block_dct

def quantizing(dct_block, big=1, alpha=1):
    if big:
        quantizer_matrix = np.ones((16, 16))
        dct_block = np.floor(np.divide(dct_block, quantizer_matrix*alpha)) * quantizer_matrix
        return dct_block
    else:
        quantizer_matrix = np.ones((8, 8))
        new_small_blocks = []
        for small_block in dct_block:  
             small_block = np.floor(np.divide(small_block, quantizer_matrix*alpha)) * quantizer_matrix * alpha
             new_small_blocks.append(small_block)
        return new_small_blocks
 
def masking_dct_and_inverse(block_dct, big=1):
    mask1 = cv2.imread("mask1.png", cv2.IMREAD_GRAYSCALE) / 255
    mask2 = cv2.imread("mask2.png", cv2.IMREAD_GRAYSCALE) /255
    if big:
        block_dct = block_dct * mask2
        return fft.idct(block_dct)
    else:
        new_small_blocks = []
        for smaller_block in block_dct:
            smaller_block = smaller_block * mask1
            new_small_blocks.append(fft.idct(smaller_block))
        return new_small_blocks

for idx in range(len(blocks_16x16)):
    if idx not in top_30percent_highest_sharpness_indices:
        blocks_16x16[idx] = masking_dct_and_inverse(quantizing(dct(blocks_16x16[idx]), alpha=alpha)) 

for idx in range(len(blocks_8x8)):
    blocks_8x8[idx] = masking_dct_and_inverse(quantizing(dct(blocks_8x8[idx], 0), 0, alpha=alpha), 0) 
       
def make_16x16_blocks(block_8x8):
    block0 = block_8x8[0]
    block1 = block_8x8[1]
    block2 = block_8x8[2]
    block3 = block_8x8[3]
    
    block_0and1_16x8 = np.concatenate((block0, block1), axis=1)
    block_2and3_16x8 = np.concatenate((block2, block3), axis=1)
    block_16x16 = np.concatenate((block_0and1_16x8, block_2and3_16x8), axis=0)
    return block_16x16
    
for idx in range(len(blocks_8x8)):
    blocks_16x16[top_30percent_highest_sharpness_indices[idx]] = make_16x16_blocks(blocks_8x8[idx])

blocks_16x16 = np.array(blocks_16x16).reshape(reversed_row_size, reversed_col_size, 16, 16)

def make_reconstructed_img(blocks_16x16):
    all_rows = []
    for i in range(len(blocks_16x16)):
        all_rows.append((np.concatenate(blocks_16x16[i], axis=1)))

    reconstructed_img = np.concatenate(all_rows, axis=0)
    return reconstructed_img

reconstructed_img = np.array(make_reconstructed_img(blocks_16x16)).clip(min=0).round()
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed Image')
plt.show()
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

print('alpha =', alpha)
ssmi = structural_similarity(img, reconstructed_img)
print('ssmi =', ssmi)
psnr = peak_signal_noise_ratio(img, reconstructed_img, data_range=255)
print('psnr =', psnr)




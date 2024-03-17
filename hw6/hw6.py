# file to run SVD compression on clown image fro hw6 big data
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from copy import deepcopy

def compress_svd(image, k):
    '''compresses the image using SVD with k singular values.'''
    U, s, V = np.linalg.svd(image, full_matrices=False)
    reconst_matrix = np.dot(U[:, :k], np.dot(np.diag(s[:k]), V[:k, :]))
    return reconst_matrix, s

if __name__ == '__main__':
    img = imageio.imread('clown.jpeg')
    img = img.mean(axis=2)  # convert to grayscale by averaging the RGB channels
    img_shuffle = deepcopy(img)
    img_shuffle = img_shuffle.flatten()
    np.random.shuffle(img_shuffle)
    img_shuffle = img_shuffle.reshape(img.shape)

    def plot_singular_vals():
        # first need plot of top 100 singular values
        _, s_shuffle = compress_svd(img_shuffle, 100)
        _, s = compress_svd(img, 100)
        plt.figure(figsize=(10, 5))
        plt.plot(s, 'r', label='Original Image')
        plt.plot(s_shuffle, 'b', label='Shuffled Image')
        plt.title('Singular Values of Clown Image')
        plt.legend()
        plt.savefig('singular_values.png')

    def plot_comparison():
        reconstr_2, _ = compress_svd(img, 2)
        reconstr_10, _ = compress_svd(img, 10)
        reconstr_20, _ = compress_svd(img, 20)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(reconstr_2, cmap='gray')
        axs[0, 1].set_title('Reconstructed Image ($k=2$)')
        axs[1, 0].imshow(reconstr_10, cmap='gray')
        axs[1, 0].set_title('Reconstructed Image ($k=10$)')
        axs[1, 1].imshow(reconstr_20, cmap='gray')
        axs[1, 1].set_title('Reconstructed Image ($k=20$)')
        plt.savefig('reconstructions.png')

    plot_singular_vals()



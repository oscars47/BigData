# second idea for mnist digits
import numpy as np
from sklearn.decomposition import PCA
from skimage.transform import resize
from scipy.ndimage import rotate
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import os
import time

THRESHOLD=128

def find_bounding_box(image):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return image[ymin:ymax+1, xmin:xmax+1]

def scale_and_convert_to_square(image, size=28):
    # Assuming `image` is the output of `find_bounding_box`
    h, w = image.shape
    scale = size / max(h, w)
    image_rescaled = resize(image, (int(h * scale), int(w * scale)), anti_aliasing=True)
    
    # Create a new square image and paste the rescaled image into it
    square_image = np.zeros((size, size))
    y_offset = (size - int(h * scale)) // 2
    x_offset = (size - int(w * scale)) // 2
    square_image[y_offset:y_offset+int(h * scale), x_offset:x_offset+int(w * scale)] = image_rescaled
    return square_image

def rotate_using_pca(image):
    # Identify non-zero (i.e., digit) pixels
    y, x = np.nonzero(image)  # Get the coordinates of non-zero pixels
    data = np.column_stack([x, y])  # Stack them as a 2D array: [[x1, y1], [x2, y2], ...]
    
    # Apply PCA on these coordinates
    pca = PCA(n_components=2)
    pca.fit(data)
    # Calculate the angle of the first principal component
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) * (180.0 / np.pi)
    
    # Rotate the image
    rotated_image = rotate(image, angle, reshape=False)  # Rotate without reshaping
    
    return rotated_image

def get_standardized(img_data, index=None, targets=None, show=False):
    # Preprocess the image
    image = img_data.reshape(28, 28)
    # apply mask
    # image = image > threshold
    bbox_image = find_bounding_box(image)
    squared_image = scale_and_convert_to_square(bbox_image)
    standardized_image = rotate_using_pca(squared_image)

    if show:
        # print out standardized image
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(image, cmap='gray')
        axs[1].imshow(squared_image, cmap='gray')
        axs[2].imshow(standardized_image, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].set_title('Squared Image')
        axs[2].set_title('Standardized Image')
        if index is not None:
            plt.savefig(f'results2/{index}.png')
        else:
            # use time to save
            timestamp = time.time()
            plt.savefig(f'results2/{timestamp}.png')
        plt.show()

    return standardized_image
    
    # # Compare with each target
    # scores = [frobenius_norm(standardized_image, target) for target in targets]
    # return np.argmin(scores)  # Return the index of the target with the smallest Frobenius norm


## comparing images ##
def frobenius_norm(A, B):
    return np.sqrt(np.sum((A - B)**2))


# create list of targets
# 8 = 26948

def load_data(save=True):
    '''load MNIST data'''
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    # separate into train/validate, test
    tv_split = int(0.8 * len(X))
    train_split = int(0.8 * tv_split)
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:tv_split], y[train_split:tv_split]
    X_test, y_test = X[tv_split:], y[tv_split:]

    if save:
        if not os.path.exists('data'):
            os.makedirs('data')
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_val.npy', y_val)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_test.npy', y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data() # run this once to save the data
    X_train = np.load('data/X_train.npy')
    # Convert the first image to a 28x28 matrix
    # index = np.random.randint(0, len(X_train))
    index = 26948
    img_data0 = np.array(X_train[index])  # Use .iloc[0] to access the first row for pandas DataFrame
    # classify_digit(img_data0, index)

    

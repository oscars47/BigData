# file to find beta, s, h vals for 0 and 1
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.linalg import svd
from skimage.transform import rotate, rescale
from scipy.ndimage import shift
import pandas as pd
from tqdm import trange, tqdm
from mnist_2 import *
from vec_angles import *

## ---- get targets for each digit ---- ##
def create_target(digit):
    '''builds sample target image for a given digit'''
    image = np.zeros((28, 28))  # Start with an all-black 28x28 image
    
    if digit == 0:
        # Making both the outer part and the inner hole fully circular
        # Outer part
        image[7:21, 10:18] = 1  # Main body of 0
        image[8:20, 9:19] = 1  # Expanding to ensure outer circular shape
        image[9:19, 8:20] = 1  # Expanding more for a smoother circular shape
        
        # Inner hole to be more circular
        image[9:19, 11:17] = 0  # Clearing the central part for the hole
        image[10:18, 10:18] = 0  # Making the hole circular
        image[11:17, 9:19] = 0  # Expanding the hole slightly for a better circular shape
        
    elif digit == 1:
        # Draw one
        image[7:21, 13:15] = 1
    elif digit == 2:
        # Draw two
        image[7:9, 10:18] = 1  # Top horizontal bar
        image[19:21, 10:18] = 1  # Bottom horizontal bar
        image[13:15, 10:18] = 1  # Middle horizontal bar
        image[9:13, 16:18] = 1  # Top right vertical bar
        image[15:19, 10:12] = 1  # Bottom left vertical bar
    elif digit == 3:
        # Draw three
        image[7:9, 10:18] = 1  # Top horizontal bar
        image[19:21, 10:18] = 1  # Bottom horizontal bar
        image[13:15, 10:18] = 1  # Middle horizontal bar
        image[9:19, 16:18] = 1  # Right vertical bar
    elif digit == 4:
        # Draw four
        image[7:15, 10:12] = 1  # Left vertical bar
        image[13:15, 10:19] = 1  # Middle horizontal bar
        image[7:21, 16:18] = 1  # Right vertical bar
    elif digit == 5:
        # Draw five
        image[7:9, 10:18] = 1  # Top horizontal bar
        image[19:21, 10:18] = 1  # Bottom horizontal bar
        image[13:15, 10:18] = 1  # Middle horizontal bar
        image[9:13, 10:12] = 1  # Top left vertical bar
        image[15:19, 16:18] = 1  # Bottom right vertical bar
    elif digit == 6:
        # Draw six
        image[7:21, 10:12] = 1  # Left vertical bar
        image[19:21, 10:18] = 1  # Bottom horizontal bar
        image[13:15, 10:18] = 1  # Middle horizontal bar
        image[15:19, 16:18] = 1  # Bottom right vertical bar
    elif digit == 7:
        # Draw seven
        image[7:9, 10:18] = 1  # Top horizontal bar
        image[9:21, 16:18] = 1  # Right vertical bar
    elif digit == 8:
        # Draw eight
        image[7:21, 10:12] = 1  # Left vertical bar
        image[7:21, 16:18] = 1  # Right vertical bar
        image[7:9, 10:18] = 1  # Top horizontal bar
        image[19:21, 10:18] = 1  # Bottom horizontal bar
        image[13:15, 10:18] = 1  # Middle horizontal bar
    elif digit == 9:
        # Draw nine
        image[7:15, 10:12] = 1  # Left vertical bar
        image[7:9, 10:18] = 1  # Top horizontal bar
        image[13:15, 10:18] = 1  # Middle horizontal bar
        image[7:21, 16:18] = 1  # Right vertical bar

    return image

def show_all_targets():
    '''displays all targets'''
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))
    for i in range(10):
        axs[i].imshow(create_target(i), cmap='gray')
        axs[i].set_title(f'Target {i}')
        axs[i].axis('off')
    plt.savefig('results_skel/targets.pdf')
    plt.show()

def do_pca(img_data, threshold=False):
    '''performs PCA on img_data'''
    # Preprocess the image
    image = img_data.reshape(28, 28)
    if threshold:
        image = image > THRESHOLD

    # Identify non-zero (i.e., digit) pixels
    y, x = np.nonzero(image)  # Get the coordinates of non-zero pixels
    data = np.column_stack([x, y])  # Stack them as a 2D array

    # Apply PCA on these coordinates
    pca = PCA(n_components=2)
    pca.fit(data)

    # Calculate the angle of the first principal component
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) * (180.0 / np.pi)

    # get the vectors
    vecs = pca.components_

    var = pca.explained_variance_ratio_

    # Rotate the image
    rotated_image = rotate(image, angle, reshape=False) 

    return rotated_image, vecs,var,  angle


# dictionary of targets
TARGS = {i: create_target(i) for i in range(10)}

TARGS_BB = {i: find_bounding_box(TARGS[i]) for i in range(10)}

TARGS_SKEL = {i: skeletonize(TARGS[i], standardize=False) for i in range(10)}

TARGS_PCA = {i: do_pca(TARGS[i]) for i in range(10)}


## ---- find disparity ---- ##

def get_disparity(img_data, target, show=False, index=None):
    '''find rotation and scaling of img relative to target (int)
    '''
    # find the bounding box
    img = img_data.reshape(28, 28)
    img = skeletonize(img, standardize=False)

    # find the bounding box of the target
    targ = TARGS[target]

    mtx1, mtx2, disparity = procrustes(img, targ)

    mtx1 = mtx1 > 0
    mtx2 = mtx2 > 0

    if show:
        print(f'Disparity to {target}:', disparity)
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(TARGS[target], cmap='gray')
        axs[1].set_title('Target Image')
        # axs[2].imshow(mtx1, cmap='gray', alpha=0.5)
        axs[2].imshow(mtx2, cmap='gray', alpha=0.5)
        axs[2].set_title('Transformed Image')
        if index is not None:
            plt.savefig(f'results_skel/{index}_{target}.pdf')
        else:
            # use time to save
            timestamp = time.time()
            plt.savefig(f'results_skel/{timestamp}.pdf')
        # plt.show()
    return disparity
    
def classify_disparity(X_data, Y_data):
    '''classify images based on disparity to targets'''
    # find the disparity for each image
    disparities = np.zeros((len(X_data), 10))
    for i in trange(len(X_data)):
        for j in range(10):
            disparities[i, j] = get_disparity(X_data[i], j)

    # classify based on disparity
    y_pred = np.argmin(disparities, axis=1)
    accuracy = np.mean(y_pred == Y_data)
    print(f'Accuracy: {accuracy:.4f}')

    # create confusion matrix
    conf_mat = np.zeros((10, 10))
    for i in range(len(Y_data)):
        conf_mat[Y_data[i], y_pred[i]] += 1

    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(conf_mat, cmap='Blues')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix, Accuracy: {accuracy:.4f}')
    plt.savefig('results_skel/confusion_matrix.pdf')

def find_bsh(img_data, target, show=False, index=None):
    '''manually find the scaling, rotation, and translation of img_data to target'''
    
    img = img_data.reshape(28, 28)
    img = skeletonize(img, standardize=False)

    targ = TARGS[target]

    # Center the shapes at the origin
    img_centroid = np.mean(img, axis=0)
    targ_centroid = np.mean(targ, axis=0)
    img_centered = img - img_centroid
    targ_centered = targ - targ_centroid

    # Normalize the shapes by their sizes
    norm_img = np.linalg.norm(img_centered, 'fro')
    norm_targ = np.linalg.norm(targ_centered, 'fro')
    X_scaled = img_centered / norm_img
    Y_scaled = targ_centered / norm_targ

    # Calculate the optimal rotation using SVD
    U, _, Vt = np.linalg.svd(np.dot(Y_scaled.T, X_scaled))
    R = np.dot(U, Vt)

    # Calculate the rotation angle
    angle_rad = np.arctan2(R[1, 0], R[0, 0])

    # Scaling factor is the ratio of norms
    scale_factor = norm_targ / norm_img

    if show:
         # Apply the scaling and rotation to the original image
        img_scaled = rescale(img, scale_factor)
        img_rotated = rotate(img_scaled, angle_rad, reshape=True)

        # Find the new centroid
        img_rotated_centroid = np.mean(img_rotated, axis=0)

        img_centroid = np.array([np.mean(np.where(img > 0), axis=1)])  # Shape (2,), [y_mean, x_mean]
        targ_centroid = np.array([np.mean(np.where(targ > 0), axis=1)])  # Shape (2,), [y_mean, x_mean]

        # Calculate the translation vector (note the inversion of axes for image coordinates)
        translation_vector = targ_centroid - img_centroid

        # Apply the translation
        # Note: shift function expects the order (y, x) for 2D images
        img_translated = shift(img_rotated, shift=[translation_vector[0]])


        # Translate the image to the target centroid
        # img_translated = img_rotated + targ_centroid

        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(targ, cmap='gray')
        axs[1].set_title('Target Image')
        axs[2].imshow(img_rotated, cmap='gray')
        axs[2].set_title('Transformed Image')
        if index is not None:
            plt.savefig(f'results_skel/{index}_{target}.pdf')
        else:
            # use time to save
            timestamp = time.time()
            plt.savefig(f'results_skel/{timestamp}.pdf')
        


    return R, scale_factor, angle_rad

def find_bsh_correct(img_data, target, show=False, index=None):
    a = img_data.reshape(28, 28)
    a = a > THRESHOLD

    a = skeletonize(a, standardize=False)


    b = TARGS[target]

    ## adapted from https://github.com/francjerez/procrustes/blob/main/procrustes.py ##
    
    # Compute centroid (ori_avg/ori_len) 
    aT = a.mean(0)	
    bT = b.mean(0)

    # Translate point cloud around own origin (0)
    A = a - aT 
    B = b - bT

    # Compute quadratic mean of point cloud
    aS = np.sum(A * A)**.5	
    bS = np.sum(B * B)**.5

    # Get scale invariant unit vector (ori/ori_rms=0_to_1_rmsd_norm_coords) 
    A /= aS	
    B /= bS

    # Compute the covariance matrix from scalar product of sets
    C = np.dot(B.T, A)

    # Decompose point cloud into orthonormal bases (U, V being U transposed as in 'np.fliplr(np.rot90(U))') and singular values (S/_)
    U, _, V = np.linalg.svd(C)

    # Compute optimal rotation matrix from unitary matrices (left U and transposed right V singular vectors)
    aR = np.dot(U, V)

	# Enforce clockwise rotation matrix by disabling svd 'best-fit' reflections (the UV matrix R determinant should always be +1)
    if np.linalg.det(aR) < 0:

        # Flip sign on second row
        V[1] *= -1

        # Rebuild rotation matrix (no need to rerun if condition skipped)
        aR = np.dot(U, V)

    # Get scale multiplier factor so we can zoom in mutable shape into reference (rotation is already computed so we can leave the norm space)
    aS = aS / bS

    # Compute translation distance from mutable shape to reference (aT - bT_fitted_to_a_space)
    aT = aT - (bT.dot(aR) * aS)

    # Rotate mutable B unto fixed A (svd helps but the method for optimal rotation matrix U is Kabsch's)  
    B_ = B.dot(aR)

    # Compute Procrustes distance from RMSD (as per Kabsch 'A-BR' instead of dummy 'A-B')
    aD = (np.sum((A - B_)**2) / len(a))**.5


    if show:
        print(f'Disparity to {target}:', aD)
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(a, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(b, cmap='gray')
        axs[1].set_title('Target Image')
        axs[2].imshow(B_, cmap='gray')
        axs[2].set_title('Transformed Image')
        if index is not None:
            plt.savefig(f'results_skel/{index}_{target}.pdf')
        else:
            # use time to save
            timestamp = time.time()
            plt.savefig(f'results_skel/{timestamp}.pdf')
        # plt.show()

def find_bsh_pca(img_data, target, show=False, index=None, return_diff_lst=False):
    '''find the scaling, rotation, and translation of img_data to target by performing PCA'''

    img_pca, img_vecs, img_var, img_angle = do_pca(img_data, threshold=False)
    targ_pca, targ_vecs, targ_var, targ_angle = do_pca(TARGS[target])

    angle_diff = img_angle - targ_angle

    # now get scale
    # ratio of the bounding boxes of the rotated images
    img_pca_t = img_pca > THRESHOLD
    targ_pca = np.where(np.isclose(targ_pca, 0), 0, 1)
    # now skeletonize
    img_pca = skeletonize(img_pca_t, standardize=False)
    # targ_pca = skeletonize(targ_pca, standardize=False)
    bbox_img = find_bounding_box(img_pca)
    bbox_targ = find_bounding_box(targ_pca)

    scale_factor_x = bbox_img.shape[1] / bbox_targ.shape[1]
    scale_factor_y = bbox_img.shape[0] / bbox_targ.shape[0]

    # scale the img bbox to the targ bbox
    bbox_img = rescale(bbox_img, (1/scale_factor_y, 1/scale_factor_x))

    # approximate the "shearing" factor as the distance from the center to the skeleton

    # fit using polar coordinates
    # get the center of the image
    y, x = np.indices(img_pca.shape)  # Get the indices for rows (y) and columns (x)
    total_mass = img_pca.sum()  # Sum of all pixel intensities

    # Calculate the weighted average of the row and column indices
    com_y = (y * img_pca).sum() / total_mass
    com_x = (x * img_pca).sum() / total_mass
    center_img =(com_x, com_y)

    # get the center of the target
    y, x = np.indices(targ_pca.shape)  # Get the indices for rows (y) and columns (x)
    total_mass = targ_pca.sum()  # Sum of all pixel intensities

    # Calculate the weighted average of the row and column indices
    com_y = (y * targ_pca).sum() / total_mass
    com_x = (x * targ_pca).sum() / total_mass
    center_targ =(com_x, com_y)

    # get the distance from the center to the skeleton
    img_skel = np.where(img_pca > 0)
    targ_skel = np.where(targ_pca > 0)
    img_dist = np.sqrt((img_skel[0] - center_img[1])**2 + (img_skel[1] - center_img[0])**2)

    img_dist_mean = np.mean(img_dist)
    targ_dist = np.sqrt((targ_skel[0] - center_targ[1])**2 + (targ_skel[1] - center_targ[0])**2)
    targ_dist_mean = np.mean(targ_dist)

    shear_factor = img_dist_mean / targ_dist_mean

    # compare thickness of img_skeleton to img_skeleton
    # get the thickness of the skeleton
    # convert from boolean to int
    img_pca_t = np.where(img_pca_t, 1, 0)
    img_pca = np.where(img_pca, 1, 0)

    img_diff = img_pca_t - img_pca # thick - skeleton
    # get avg distance from center to skeleton
    img_diff_skel = np.where(img_diff > 0)
    img_diff_dist = np.sqrt((img_diff_skel[0] - center_img[1])**2 + (img_diff_skel[1] - center_img[0])**2)
    img_diff_dist_mean = np.mean(img_diff_dist)
    

    if show:

        print(f'Image angle: {img_angle}')
        print(f'Targ angle: {targ_angle}')
        print(f'Angle diff: {angle_diff}')
        print(f'Scale factor x: {scale_factor_x}')
        print(f'Scale factor y: {scale_factor_y}')
        print(f'Shear factor: {shear_factor}')
        print(f'Img diff dist: {img_diff_dist_mean}')

        # plot comparison of original img data, targ, with pca vecs
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        axs[0][0].imshow(img_data.reshape(28, 28), cmap='gray')
        # plot pca vecs. make sure they start at the center of the image

        colors = ['r', 'b']

        for i, component in enumerate(img_vecs):
        # Direction for the PCA vector
            dx, dy = component[0],component[1]  
            # Plot the vector from the center with scaling
            scale = 10 * img_var[i]/img_var[0]
            axs[0][0].quiver(center_img[0], center_img[1], dx, dy, scale=scale, color=colors[i])

        axs[0][0].set_title('Original Image')
        axs[0][1].imshow(TARGS[target], cmap='gray')
        
        for i, component in enumerate(targ_vecs):
        # Direction for the PCA vector
            dx, dy = component[0],component[1]  
            # Plot the vector from the center with scaling
            scale = 10 * targ_var[i]/targ_var[0]
            axs[0][1].quiver(center_targ[0], center_targ[1], dx, dy, scale=scale, color=colors[i])

        axs[0][1].set_title('Target Image')
        axs[1][0].imshow(img_pca, cmap='gray')
        axs[1][0].set_title('PCA Image')
        axs[1][1].imshow(targ_pca, cmap='gray')
        axs[1][1].set_title('PCA Target Image')
        # plot bounding boxes
        axs[2][0].imshow(bbox_img, cmap='gray')
        axs[2][0].set_title('Bounding Box Image')
        axs[2][1].imshow(bbox_targ, cmap='gray')
        axs[2][1].set_title('Bounding Box Target')
        plt.savefig(f'results_skel/pca_{index}_{target}.pdf')
        
        # new figure for distance
        plt.figure(figsize=(10, 5))
        plt.plot(img_dist, label='Image')
        plt.plot(targ_dist, label='Target')
        plt.plot(img_diff_dist, label='Diff')
        plt.legend()
        plt.savefig(f'results_skel/distances_{index}_{target}.pdf')

    if not return_diff_lst:
        return angle_diff, scale_factor_x, scale_factor_y, shear_factor, img_diff_dist_mean
    else:
        return angle_diff, scale_factor_x, scale_factor_y, shear_factor, img_diff_dist_mean, img_diff_dist, img_dist, targ_dist

def find_bsh_pca_only_img(img_data, targ_angle, targ_dist_mean, bbox_targ):
    img_pca, img_vecs, img_var, img_angle = do_pca(img_data, threshold=False)

    angle_diff = img_angle - targ_angle

    # now get scale
    # ratio of the bounding boxes of the rotated images
    img_pca_t = img_pca > THRESHOLD
    img_pca = skeletonize(img_pca_t, standardize=False)
    # targ_pca = skeletonize(targ_pca, standardize=False)
    bbox_img = find_bounding_box(img_pca)

    scale_factor_x = bbox_img.shape[1] / bbox_targ.shape[1]
    scale_factor_y = bbox_img.shape[0] / bbox_targ.shape[0]

    # scale the img bbox to the targ bbox
    bbox_img = rescale(bbox_img, (1/scale_factor_y, 1/scale_factor_x))

    # approximate the "shearing" factor as the distance from the center to the skeleton

    # fit using polar coordinates
    # get the center of the image
    y, x = np.indices(img_pca.shape)  # Get the indices for rows (y) and columns (x)
    total_mass = img_pca.sum()  # Sum of all pixel intensities

    # Calculate the weighted average of the row and column indices
    com_y = (y * img_pca).sum() / total_mass
    com_x = (x * img_pca).sum() / total_mass
    center_img =(com_x, com_y)

    # get the distance from the center to the skeleton
    img_skel = np.where(img_pca > 0)
    img_dist = np.sqrt((img_skel[0] - center_img[1])**2 + (img_skel[1] - center_img[0])**2)

    img_dist_mean = np.mean(img_dist)

    shear_factor = img_dist_mean / targ_dist_mean

    # compare thickness of img_skeleton to img_skeleton
    # get the thickness of the skeleton
    # convert from boolean to int
    img_pca_t = np.where(img_pca_t, 1, 0)
    img_pca = np.where(img_pca, 1, 0)

    img_diff = img_pca_t - img_pca # thick - skeleton
    # get avg distance from center to skeleton
    img_diff_skel = np.where(img_diff > 0)
    img_diff_dist = np.sqrt((img_diff_skel[0] - center_img[1])**2 + (img_diff_skel[1] - center_img[0])**2)
    img_diff_dist_mean = np.mean(img_diff_dist)

    return angle_diff, scale_factor_x, scale_factor_y, shear_factor, img_diff_dist, img_dist, img_diff_dist_mean, img_dist_mean, img_pca_t, img_pca, bbox_img

def comp_bsh_pca(X_data, target):
    '''compare the bsh of each img in X_data to specified target using pca'''

    # plot comparison of original img data, targ, with pca vecs
    targ_pca, targ_vecs, targ_var, targ_angle = do_pca(TARGS[target])
    targ_pca = np.where(np.isclose(targ_pca, 0), 0, 1)
    bbox_targ = find_bounding_box(targ_pca)

    # get the center of the target
    y, x = np.indices(targ_pca.shape)  # Get the indices for rows (y) and columns (x)
    total_mass = targ_pca.sum()  # Sum of all pixel intensities

    # Calculate the weighted average of the row and column indices
    com_y = (y * targ_pca).sum() / total_mass
    com_x = (x * targ_pca).sum() / total_mass
    center_targ =(com_x, com_y)

    # get the distance from the center to the skeleton
    targ_skel = np.where(targ_pca > 0)
    targ_dist = np.sqrt((targ_skel[0] - center_targ[1])**2 + (targ_skel[1] - center_targ[0])**2)
    targ_dist_mean = np.mean(targ_dist)

    # create df to store results
    results = pd.DataFrame(columns=['angle_diff', 'scale_factor_x', 'scale_factor_y', 'shear_factor', 'img_diff_dist_mean', 'img_dist_mean'])

    # separate folder to store the distances
    if not os.path.exists('results_skel/dist'):
        os.makedirs('results_skel/dist')

    # now plot the left half
    for i in trange(len(X_data)):
        img_data = X_data[i]

        angle_diff, scale_factor_x, scale_factor_y, shear_factor, img_diff_dist, img_dist, img_diff_dist_mean, img_dist_mean, img_orig, img_pca, bbox_img = find_bsh_pca_only_img(img_data, targ_angle, targ_dist_mean, bbox_targ)

        results = pd.concat([results, pd.DataFrame({'angle_diff': [angle_diff], 'scale_factor_x': [scale_factor_x], 'scale_factor_y': [scale_factor_y], 'shear_factor': [shear_factor], 'img_diff_dist_mean': [img_diff_dist_mean], 'img_dist_mean': [img_dist_mean]})], ignore_index=True)

        # save the distances
        np.save(f'results_skel/dist/img_diff_ls_{i}_{target}.npy', img_diff_dist)
        np.save(f'results_skel/dist/img_ls_{i}_{target}.npy', img_dist)
        np.save(f'results_skel/dist/img_orig_{i}_{target}.npy', img_orig)
        np.save(f'results_skel/dist/img_pca_{i}_{target}.npy', img_pca)
        np.save(f'results_skel/dist/bbox_img_{i}_{target}.npy', bbox_img)

    # save results
    results.to_csv(f'results_skel/pca_results_{target}_{len(X_data)}.csv')


def plot_comp_bsh_pca(pca_result_path_ls=['results_skel/pca_results_0_4407.csv', 'results_skel/pca_results_1_5109.csv'], dist_path='results_skel/dist'):
    '''plot the results of comp_bsh_pca'''

    # initialize dict to store results
    pca_results_dict = {}

    # load the results
    for pca_result_path in pca_result_path_ls:
        # get target
        target = pca_result_path.split('_')[-2]
        results = pd.read_csv(pca_result_path)
        pca_results_dict[target] = results

    # for each target, plot the results
    def plot_dist(target):
        '''plot the results'''

        # get the target info
        results = pca_results_dict[target]

        # plot this
        # angle diff, scale_factor_x, scale_factor_y, shear_factor, img_diff_dist_mean, img_dist_mean
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0][0].hist(results['angle_diff'], bins=100)
        axs[0][0].set_title('Angle Diff')
        axs[0][1].hist(results['scale_factor_x'], bins=100)
        axs[0][1].set_title('Scale Factor X')
        axs[0][2].hist(results['scale_factor_y'], bins=100)
        axs[0][2].set_title('Scale Factor Y')
        axs[1][0].hist(results['shear_factor'], bins=100)
        axs[1][0].set_title('Shear Factor')
        axs[1][1].hist(results['img_diff_dist_mean'], bins=100)
        axs[1][1].set_title('Img Diff Dist Mean')
        axs[1][2].hist(results['img_dist_mean'], bins=100)
        axs[1][2].set_title('Img Dist Mean')
        plt.savefig(f'results_skel/pca_results_{target}.pdf')

        # then plot of img_list, img_diff_list
        dist_files = os.listdir(dist_path)
        img_list = [np.load(f'{dist_path}/{f}') for f in dist_files if f.startswith(f'img_ls') and f.endswith(f'{target}.npy')]
        img_diff_list = [np.load(f'{dist_path}/{f}') for f in dist_files if f.startswith(f'img_diff_ls') and f.endswith(f'{target}.npy')]

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        for i in range(len(img_list)):
            axs[0].plot(img_list[i], label=f'img_{i}')
            axs[1].plot(img_diff_list[i], label=f'img_diff_{i}')
        axs[0].set_title('Img List')
        axs[1].set_title('Img Diff List')
        plt.savefig(f'results_skel/dist_results_{target}.pdf')

        # then plot of before and after pca images
        img_orig_list = [np.load(f'{dist_path}/{f}') for f in dist_files if f.startswith(f'img_orig') and f.endswith(f'{target}.npy')]
        img_pca_list = [np.load(f'{dist_path}/{f}') for f in dist_files if f.startswith(f'img_pca') and f.endswith(f'{target}.npy')]
        bbox = [np.load(f'{dist_path}/{f}') for f in dist_files if f.startswith(f'bbox_img') and f.endswith(f'{target}.npy')]

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        for i in range(len(img_orig_list)):
            axs[0].imshow(img_orig_list[i], cmap='gray', alpha=0.5)
            axs[1].imshow(img_pca_list[i], cmap='gray', alpha=0.5)
        axs[0].set_title('Original Image')
        axs[1].set_title('PCA Image')
        plt.savefig(f'results_skel/img_results_{target}.pdf')

        # plot the bounding boxes
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(len(bbox)):
            axs.imshow(bbox[i], cmap='gray', alpha=0.5)
        axs.set_title('Bounding Box Image')
        plt.savefig(f'results_skel/bbox_results_{target}.pdf')

    for target in pca_results_dict.keys():
        plot_dist(target)

# compare all 0s vs 1s
def compare_0_1(zeros, ones):
    ''' compare 0s and 1s, default, skeleton'''
    
    # get skeletons
    zeros_skels = [skeletonize(z, standardize=False) for z in zeros]

    ones_skels = [skeletonize(o, standardize=False) for o in ones]

    # overplot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.mean(zeros_skels, axis=0), cmap='gray')
    axs[0].set_title('Mean 0')
    axs[1].imshow(np.mean(ones_skels, axis=0), cmap='gray')
    axs[1].set_title('Mean 1')
    # get number of data points
    num_zeros = len(zeros)
    num_ones = len(ones)
    plt.savefig(f'results_skel/mean_0_1_{num_zeros}_{num_ones}.pdf')

    

if __name__ == '__main__':
    # Load the data
    if not os.path.exists('results_skel'):
        os.makedirs('results_skel')
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_train = y_train.astype(int)

    zeros = X_train[y_train == 0]
    ones = X_train[y_train == 1]
    twos = X_train[y_train == 2]
    threes = X_train[y_train == 3]
    fours = X_train[y_train == 4]
    fives = X_train[y_train == 5]
    sixes = X_train[y_train == 6]
    sevens = X_train[y_train == 7]
    eights = X_train[y_train == 8]
    nines = X_train[y_train == 9]

    # select first 100 from each and combine
    mini_total = np.concatenate((zeros[:100], ones[:100], twos[:100], threes[:100], fours[:100], fives[:100], sixes[:100], sevens[:100], eights[:100], nines[:100]), axis=0)
    mini_total_targs = np.concatenate([np.repeat(i, 100) for i in range(10)])


    classify_disparity(mini_total, mini_total_targs)
    # compare_0_1(zeros[:100], ones[:100])
    # for i in trange(100):
    #     find_bsh_correct(zeros[i], 0, show=True, index=i)
    #     find_bsh_correct(ones[i], 1, show=True, index=i)
    # find_bsh_pca(zeros[1], 0, show=True, index=1)
    # find_bsh_pca(ones[1], 1, show=True, index=1)
    # comp_bsh_pca(zeros, 0)
    # comp_bsh_pca(ones, 1)
    # plot_comp_bsh_pca()

# plot the digits in polar coordinates

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import trange
from mnist_2 import *

def cartesian_to_polar(x, y, center):
    '''Convert Cartesian coordinates to polar (radius and angle).'''
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y - center[1], x - center[0])
    return r, theta

def convert_img_polar(img_data, index):
    '''Plots image into polar coordinates from the center'''

    # Preprocess the image
    processed = get_standardized(img_data, index)


    # get center which is weighted average of the non-zero pixels
    ys, xs = np.nonzero(processed)  # Get indices of non-zero (true) elements
    weighted_y = np.mean(ys)
    weighted_x = np.mean(xs)
    
    # Convert the image to polar coordinates
    r_ls = []
    theta_ls = []

    for i in range(processed.shape[0]):
        for j in range(processed.shape[1]):
            if processed[i, j] > 0:
                if processed[i, j] > THRESHOLD:
                    r, theta = cartesian_to_polar(i, j, (weighted_x, weighted_y))
                    r_ls.append(r)
                    theta_ls.append(theta)

    # Plot the standardized image and its polar representation
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(processed, cmap='gray')
    # plot the center
    axs[0].scatter(weighted_x, weighted_y, s=5, c='red')
    axs[0].set_title('Standardized Image')
    axs[1].scatter(theta_ls, r_ls, s=1, c='black')
    axs[1].set_title('Polar Representation')
    plt.savefig(f'results2/polar_{index}.png')
    # plt.show()

def find_sig_max_density_x(processed, xs, threshold=3, show_plot=False):
    '''
    Find the x value with the maximum density of pixels that is significantly higher than the lowest
    
    Args:
    xs (numpy.ndarray): The x coordinates of the active pixels.
    threshold (int): The minimum difference in pixel count required to consider the density significant.
    show_plot (bool): Whether to show a histogram of the x coordinates.
    
    Returns:
    int: The x value with the significant maximum density, or the center of the image if no significant difference is found.
    '''
    # Step 1: Compute the histogram of x coordinates of active pixels
    x_hist, x_bins = np.histogram(xs, bins=np.arange(processed.shape[1]+1))
    
    # Sort the histogram values in descending order while preserving indices
    sorted_indices = np.argsort(-x_hist)
    max_density_x = sorted_indices[0]
    lowest_density_x = sorted_indices[-1]
    
    # Check if the difference between the highest and lowest is above the threshold
    if x_hist[max_density_x] - x_hist[lowest_density_x] > threshold:
        max_density =  max_density_x
    else:
        # return the center of the image
        max_density =  processed.shape[1] // 2

    # Plot the histogram for visualization
    if show_plot:
        plt.hist(xs, bins=np.arange(processed.shape[1]+1))
        plt.axvline(max_density, color='red')
        plt.show()

    return max_density

def find_cut_fit_line(img_data, show_plot = False):
    '''find the cut and fit lines for the image.

    Params:
        img_data (numpy.ndarray): The image data.
        show_plot (bool): Whether to show the plots.

    Returns:
        list: the models of the fits.
        list: the chi2red values of the fits.

    '''

    # Preprocess the image
    processed = get_standardized(img_data)
    # make binary
    processed = processed > THRESHOLD

    # get coordinates of non-zero pixels
    ys, xs = np.nonzero(processed)  # Get indices of non-zero (true) elements
    # get the outline of the digit

    if show_plot:
        plt.scatter(xs, ys, color='red')
        plt.imshow(processed, cmap='gray', origin='lower')
        plt.show()

    # Step 1: Compute the histogram of x coordinates of active pixels
    max_density_x = find_sig_max_density_x(processed, xs, show_plot=show_plot)
    if show_plot:
        print('Max density x:', max_density_x)

    # Define the regions to the left and right of the cut
    left_region_filter = xs < max_density_x
    right_region_filter = xs > max_density_x

    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Prepare subplots

    models = []
    chi2reds = []

    for index, (region_filter, ax_img, ax_res) in enumerate(zip(
        [left_region_filter, right_region_filter], axes[0], axes[1])):
        
        region_x_coords = xs[region_filter]
        region_y_coords = ys[region_filter]

        if len(region_x_coords) == 0:  # Check if there are no points in the region
            continue
        
        # Linear regression
        X = region_x_coords.reshape(-1, 1)
        y = region_y_coords
        model = LinearRegression().fit(X, y)
        
        # Calculate normalized residuals and chi2red
        residuals = y - model.predict(X)
        chi2 = np.sum((residuals / y)**2)
        chi2red = chi2 / (len(y) - 2)
        
        # Store results
        models.append((model.coef_[0], model.intercept_))
        chi2reds.append(chi2red)

        if show_plot:
            # Plot image and fit as well as residuals
            ax_img.imshow(processed, cmap='gray', origin='lower')
            ax_img.scatter(region_x_coords, region_y_coords, color='red')
            line_x = np.array([region_x_coords.min(), region_x_coords.max()])
            line_y = model.coef_[0] * line_x + model.intercept_
            ax_img.plot(line_x, line_y, color='blue', linewidth=2)
            ax_img.set_xlim([0, processed.shape[1]])
            ax_img.set_ylim([0, processed.shape[0]])
            ax_img.set_title(f'Line {index+1}, $\\chi^2_\\nu = {chi2red:.2f}$')
            
            ax_res.scatter(region_x_coords, residuals, color='red')
            ax_res.axhline(0, color='black', linewidth=1)
            ax_res.set_xlim([0, processed.shape[1]])
            ax_res.set_ylim([min(residuals), max(residuals)])
            ax_res.set_title(f'Residuals Line {index+1}')

    if show_plot:
        plt.tight_layout()
        plt.show()
    return models, chi2reds

def benchmark_0_1(X_train, y_train, num_times=1000):
    '''run cut and fit process num_times for 0s and 1s and compare the chi2red distributions.'''
    # get only 0s
    zeros = X_train[y_train == 0]
    # get only 1s
    ones = X_train[y_train == 1]
    
    chi2red_zeroL = []
    chi2red_zeroR = []
    chi2red_oneL = []
    chi2red_oneR = []
    for i in trange(num_times):
        try:
            _, chi2reds = find_cut_fit_line(zeros[i])
            chi2red_zeroL.append(chi2reds[0])
            chi2red_zeroR.append(chi2reds[1])
        except:
            pass

        try:
            _, chi2reds = find_cut_fit_line(ones[i])
            chi2red_oneL.append(chi2reds[0])
            chi2red_oneR.append(chi2reds[1])
        except:
            pass

    # remove infinite values
    chi2red_zeroL = [x for x in chi2red_zeroL if x != np.inf]
    chi2red_zeroR = [x for x in chi2red_zeroR if x != np.inf]
    chi2red_oneL = [x for x in chi2red_oneL if x != np.inf]
    chi2red_oneR = [x for x in chi2red_oneR if x != np.inf]

    # create total histogram 2,2
    # calculate avg and sem for each
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].hist(chi2red_zeroL, bins=20)
    zero_L_avg = np.mean(chi2red_zeroL)
    zero_L_sem = np.std(chi2red_zeroL) / np.sqrt(len(chi2red_zeroL))
    axs[0, 0].set_title(f'Zero Left: $\\chi^2_\\nu = {zero_L_avg:.2f} \\pm {zero_L_sem:.2f}$')
    axs[0, 1].hist(chi2red_zeroR, bins=20)
    zero_R_avg = np.mean(chi2red_zeroR)
    zero_R_sem = np.std(chi2red_zeroR) / np.sqrt(len(chi2red_zeroR))
    axs[0, 1].set_title(f'Zero Right: $\\chi^2_\\nu = {zero_R_avg:.2f} \\pm {zero_R_sem:.2f}$')
    axs[1, 0].hist(chi2red_oneL, bins=20)
    one_L_avg = np.mean(chi2red_oneL)
    one_L_sem = np.std(chi2red_oneL) / np.sqrt(len(chi2red_oneL))
    axs[1, 0].set_title(f'One Left: $\\chi^2_\\nu = {one_L_avg:.2f} \\pm {one_L_sem:.2f}$')
    axs[1, 1].hist(chi2red_oneR, bins=20)
    one_R_avg = np.mean(chi2red_oneR)
    one_R_sem = np.std(chi2red_oneR) / np.sqrt(len(chi2red_oneR))
    axs[1, 1].set_title(f'One Right: $\\chi^2_\\nu = {one_R_avg:.2f} \\pm {one_R_sem:.2f}$')

    plt.tight_layout()
    plt.savefig(f'results2/chi2red_comp_{num_times}.png')


def benchmark_all(X_train, y_train, num_times=1000):
    '''run cut and fit process num_times for all digits and compare the chi2red distributions.'''
    
    # divide up the data by digit
    data = {i: X_train[y_train == i] for i in range(10)}

    # create empty lists for each digit
    chi2reds_L = {i: [] for i in range(10)}
    chi2reds_R = {i: [] for i in range(10)}

    for i in trange(num_times):
        for digit in range(10):
            try:
                _, chi2reds = find_cut_fit_line(data[digit][i])
                chi2reds_L[digit].append(chi2reds[0])
                chi2reds_R[digit].append(chi2reds[1])
            except:
                pass

    # remove infinite values
    for digit in range(10):
        chi2reds_L[digit] = [x for x in chi2reds_L[digit] if x != np.inf]
        chi2reds_R[digit] = [x for x in chi2reds_R[digit] if x != np.inf]

    # create total histogram of 10*2 = 20 total panels, organized 5 rows, 4 columns
    # calculate avg and sem for each
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for digit in range(10):
        row = digit // 2
        col = digit % 2
        axs[row, col].hist(chi2reds_L[digit], bins=20)
        avg = np.mean(chi2reds_L[digit])
        sem = np.std(chi2reds_L[digit]) / np.sqrt(len(chi2reds_L[digit]))
        axs[row, col].set_title(f'{digit} Left: $\\chi^2_\\nu = {avg:.2f} \\pm {sem:.2f}$')
        axs[row, col+2].hist(chi2reds_R[digit], bins=20)
        avg = np.mean(chi2reds_R[digit])
        sem = np.std(chi2reds_R[digit]) / np.sqrt(len(chi2reds_R[digit]))
        axs[row, col+2].set_title(f'{digit} Right: $\\chi^2_\\nu = {avg:.2f} \\pm {sem:.2f}$')

    plt.tight_layout()
    plt.savefig(f'results2/chi2red_comp_all_{num_times}.png')

if __name__ == '__main__':
    if not os.path.exists('results2'):
        os.makedirs('results2')
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data() # run this once to save the data
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_train = y_train.astype(int)

    

    # find_cut_and_fit_line(zeros[0], show_plot=True)
    # find_cut_and_fit_line(ones[0], show_plot=True)

    benchmark_all(X_train, y_train, 1000)
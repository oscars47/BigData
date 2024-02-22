# plot the digits in polar coordinates

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import trange
from mnist_2 import *
import json

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

def perform_linear_regression(region_x_coords, region_y_coords):
    '''
    Perform linear regression on the provided region coordinates.
    '''
    X = region_x_coords.reshape(-1, 1)
    y = region_y_coords
    model = LinearRegression().fit(X, y)
    
    # Calculate normalized residuals and chi2
    residuals = y - model.predict(X)
    chi2 = np.sum((residuals)**2)
    
    return model, chi2, residuals

def find_cut_fit_line_multiple(img_data, max_cuts=1, show_plot=False):
    '''
    
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

    ys, xs = np.nonzero(processed)  # Get indices of non-zero (true) elements
    
    if show_plot:
        plt.scatter(xs, ys, color='red')
        plt.imshow(processed, cmap='gray', origin='lower')
        plt.show()

    models = []
    chi2reds = []
    residuals = []
    cuts = 0
    while cuts < max_cuts:
        max_density_x = find_sig_max_density_x(processed, xs, show_plot=show_plot)
        left_region_filter = xs < max_density_x
        right_region_filter = xs > max_density_x

        # Perform regression on the left region
        left_region_x_coords = xs[left_region_filter]
        left_region_y_coords = ys[left_region_filter]
        if len(left_region_x_coords) > 1:  # Need at least 2 points to fit
            model, chi2red, residual = perform_linear_regression(left_region_x_coords, left_region_y_coords)
            models.append(model)
            chi2reds.append(chi2red)
            residuals.append(residual)
        
        # Update xs and ys to only include the right region for the next iteration
        xs = xs[right_region_filter]
        ys = ys[right_region_filter]
        
        cuts += 1

    return models, chi2reds

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
    chi2s = []
    if show_plot:
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
            chi2 = np.sum((residuals)**2)
            # chi2red = chi2 / (len(y) - 2)
            
            # Store results
            models.append((model.coef_[0], model.intercept_))
            chi2s.append(chi2)

            

    else:
        for index, region_filter in enumerate(
            [left_region_filter, right_region_filter]):
            
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
            chi2 = np.sum((residuals)**2)
            # chi2red = chi2 / (len(y) - 2)
            
            # Store results
            models.append((model.coef_[0], model.intercept_))
            chi2s.append(chi2)
        
    return models, chi2s

def benchmark_0_1(X_train, y_train, num_times=1000):
    '''run cut and fit process num_times for 0s and 1s and compare the chi2red distributions.'''
    # get only 0s
    zeros = X_train[y_train == 0]
    # get only 1s
    ones = X_train[y_train == 1]
    
    chi2_zeroL = []
    chi2_zeroR = []
    chi2_oneL = []
    chi2_oneR = []
    for i in trange(num_times):
        try:
            _, chi2s = find_cut_fit_line(zeros[i])
            chi2_zeroL.append(chi2s[0])
            chi2_zeroR.append(chi2s[1])
        except:
            pass

        try:
            _, chi2reds = find_cut_fit_line(ones[i])
            chi2_oneL.append(chi2s[0])
            chi2_oneR.append(chi2s[1])
        except:
            pass

    # remove infinite values
    chi2_zeroL = [x for x in chi2_zeroL if x != np.inf]
    chi2_zeroR = [x for x in chi2_zeroR if x != np.inf]
    chi2_oneL = [x for x in chi2_oneL if x != np.inf]
    chi2_oneR = [x for x in chi2_oneR if x != np.inf]

    # create total histogram 2,2
    # calculate avg and sem for each
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].hist(chi2_zeroL, bins=20)
    zero_L_avg = np.mean(chi2_zeroL)
    zero_L_sem = np.std(chi2_zeroL) / np.sqrt(len(chi2_zeroL))
    axs[0, 0].set_title(f'Zero Left: $\\chi^2_\\nu = {zero_L_avg:.2f} \\pm {zero_L_sem:.2f}$')
    axs[0, 1].hist(chi2_zeroR, bins=20)
    zero_R_avg = np.mean(chi2_zeroR)
    zero_R_sem = np.std(chi2_zeroR) / np.sqrt(len(chi2_zeroR))
    axs[0, 1].set_title(f'Zero Right: $\\chi^2_\\nu = {zero_R_avg:.2f} \\pm {zero_R_sem:.2f}$')
    axs[1, 0].hist(chi2_oneL, bins=20)
    one_L_avg = np.mean(chi2_oneL)
    one_L_sem = np.std(chi2_oneL) / np.sqrt(len(chi2_oneL))
    axs[1, 0].set_title(f'One Left: $\\chi^2_\\nu = {one_L_avg:.2f} \\pm {one_L_sem:.2f}$')
    axs[1, 1].hist(chi2_oneR, bins=20)
    one_R_avg = np.mean(chi2_oneR)
    one_R_sem = np.std(chi2_oneR) / np.sqrt(len(chi2_oneR))
    axs[1, 1].set_title(f'One Right: $\\chi^2_\\nu = {one_R_avg:.2f} \\pm {one_R_sem:.2f}$')

    plt.tight_layout()
    plt.savefig(f'results2/chi2_comp_{num_times}.png')

def benchmark_all(X_train, y_train, num_times=1000, cut_multiple=False):
    '''run cut and fit process num_times for all digits and compare the chi2 distributions.

    Args:
        X_train (numpy.ndarray): The training data.
        y_train (numpy.ndarray): The training labels.
        num_times (int): The number of times to run the process.
        cut_multiple (bool): Whether to use the multiple cut method.
    
    '''

    if cut_multiple:
        fcfl = find_cut_fit_line_multiple
    else:
        fcfl = find_cut_fit_line
    
    # divide up the data by digit
    data = {i: X_train[y_train == i] for i in range(10)}

    # create histrogram of number of samples for each digit
    fig, ax = plt.subplots()
    ax.bar(data.keys(), [len(data[i]) for i in range(10)])
    ax.set_title('Number of samples for each digit')
    plt.savefig(f'results2/num_samples_{num_times}_{cut_multiple}.png')

    # create empty lists for each digit
    chi2s_L = {i: [] for i in range(10)}
    chi2s_R = {i: [] for i in range(10)}
    chi2s_combined = {i: [] for i in range(10)}

    for i in trange(num_times):
        for digit in range(10):
            _, chi2reds = fcfl(data[digit][i])
            if len(chi2reds) ==2:
                chi2s_L[digit].append(chi2reds[0])
                chi2s_R[digit].append(chi2reds[1])
            # append to combined
            for chi2red in chi2reds:
                chi2s_combined[digit].append(chi2red)
            # chi2s_combined[digit].append(np.mean(chi2reds))
               
            # except:
            #     print(f'Error with digit {digit} and index {i}, length of chi2s_L: {len(chi2s_L[digit])}, length of chi2s_R: {len(chi2s_R[digit])}')

    # remove infinite values
    for digit in range(10):
        chi2s_L[digit] = [x for x in chi2s_L[digit] if x != np.inf]
        chi2s_R[digit] = [x for x in chi2s_R[digit] if x != np.inf]

    # create total histogram of 10*2 = 20 total panels, organized 5 rows, 4 columns
    # calculate avg and sem for each
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for digit in range(0,10,2):
        row = digit // 2
        col = digit % 2
        axs[row, col].hist(chi2s_L[digit], bins=20)
        avg = np.mean(chi2s_L[digit])
        sem = np.std(chi2s_L[digit]) / np.sqrt(len(chi2s_L[digit]))
        axs[row, col].set_title(f'{digit} Left: $\\chi^2_\\nu = {avg:.2f} \\pm {sem:.2f}$')
        axs[row, col+1].hist(chi2s_R[digit], bins=20)
        avg = np.mean(chi2s_R[digit])
        sem = np.std(chi2s_R[digit]) / np.sqrt(len(chi2s_R[digit]))
        axs[row, col+1].set_title(f'{digit} Right: $\\chi^2_\\nu = {avg:.2f} \\pm {sem:.2f}$')
    
    plt.tight_layout()
    plt.savefig(f'results2/chi2_comp_all_{num_times}_{cut_multiple}.png')

     # save the chi2red dict as .json
    with open(f'results2/chi2_{num_times}.json', 'w') as f:
        json.dump({'chi2s_L': chi2s_L, 'chi2s_R': chi2s_R}, f)

    # save the average and sem for each digit using L and R
    with open(f'results2/chi2_avg_sem_{num_times}_{cut_multiple}.json', 'w') as f:
        json.dump({'chi2_combined': {i: (np.mean(chi2s_combined[i]), np.std(chi2s_combined[i]) / np.sqrt(len(chi2s_combined[i]))) for i in range(10)},    
            'chi2s_L': {i: (np.mean(chi2s_L[i]), np.std(chi2s_L[i]) / np.sqrt(len(chi2s_L[i]))) for i in range(10)}, 
                   'chi2s_R': {i: (np.mean(chi2s_R[i]), np.std(chi2s_R[i]) / np.sqrt(len(chi2s_R[i]))) for i in range(10)}}, f)
        
    # plot for each digit the combined chi2red mean and sem to see how they overlap
    fig, ax = plt.subplots()
    x = np.arange(10)
    avg = [np.mean(chi2s_combined[i]) for i in range(10)]
    sem = [np.std(chi2s_combined[i]) / np.sqrt(len(chi2s_combined[i])) for i in range(10)]
    # # make points really small
    # ax.errorbar(x, avg, yerr=sem, fmt='o', markersize=1, capsize=5, elinewidth=1, markeredgewidth=1, color='black')
    # put horizontal lines at mean + sem and mean - sem
    # get color for each digit
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    for i in range(10):
        ax.hlines(avg[i] + sem[i], 0, 9, color=colors[i])
        ax.hlines(avg[i] - sem[i], 0, 9, color=colors[i])
        ax.hlines(avg[i] + 3*sem[i], 0, 9, color=colors[i], linestyles='--')
        ax.hlines(avg[i] - 3*sem[i], 0, 9, color=colors[i], linestyles='--')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_title('Combined $\chi^2$ for each digit')
    plt.savefig(f'results2/chi2_combined_{num_times}_{cut_multiple}.png')

def predict_0_1(X_test, y_test, combined_chi2_path='results2/chi2_avg_sem_4029.json'):
    '''predict 0 vs 1 using the chi2 values.'''

    # get dict of chi2s
    with open(combined_chi2_path, 'r') as f:
        chi2s = json.load(f)

    # separate into 0s and 1s
    zero_or_one = X_test[(y_test == 0) | (y_test == 1)]
    y_test = y_test[(y_test == 0) | (y_test == 1)]

    y_pred = []
    for i in trange(len(zero_or_one)):
        _, chi2reds = find_cut_fit_line(zero_or_one[i])
        chi2red_comb = np.mean(chi2reds)

        # find the digit with the smallest difference
        min_diff = np.inf
        min_digit = None
        for digit in [0, 1]:
            diff = abs(chi2red_comb - chi2s['chi2_combined'][str(digit)][0])
            if diff < min_diff:
                min_diff = diff
                min_digit = digit
        y_pred.append(min_digit)

    # compute accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy}')

    # create confusion matrix
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_test)):
        confusion_matrix[y_test[i], y_pred[i]] += 1

    # plot confusion matrix
    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix, cmap='viridis')

    # We want to show all ticks...
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    # ... and label them with the respective list entries
    ax.set_xticklabels([str(i) for i in range(2)])
    ax.set_yticklabels([str(i) for i in range(2)])

    # create text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(confusion_matrix[i, j]),
                           ha="center", va="center", color="w")
            
    ax.set_title(f"Confusion Matrix, Acc = {accuracy:.2f}")
    fig.tight_layout()
    plt.savefig('results2/confusion_matrix_0_1.png')

def predict_all(X_test, y_test, combined_chi2_path='results2/chi2_avg_sem_4029.json'):
    '''predict the digits using the chi2 values.'''
    # get dict of chi2s
    with open(combined_chi2_path, 'r') as f:
        chi2s = json.load(f)

    y_pred = []
    for i in trange(len(X_test)):
        _, chi2reds = find_cut_fit_line(X_test[i])
        chi2red_comb = np.mean(chi2reds)

        # find the digit with the smallest difference
        min_diff = np.inf
        min_digit = None
        for digit in range(10):
            diff = abs(chi2red_comb - chi2s['chi2_combined'][str(digit)][0])
            if diff < min_diff:
                min_diff = diff
                min_digit = digit
        y_pred.append(min_digit)

    # compute accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy}')

    # create confusion matrix
    confusion_matrix = np.zeros((10, 10))
    for i in range(len(y_test)):
        confusion_matrix[y_test[i], y_pred[i]] += 1

    # plot confusion matrix
    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix, cmap='viridis')

    # We want to show all ticks...
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    # ... and label them with the respective list entries
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_yticklabels([str(i) for i in range(10)])

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
    #          rotation_mode="anchor")
    
    # create text annotations
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(confusion_matrix[i, j]),
                           ha="center", va="center", color="w")
            
    ax.set_title(f"Confusion Matrix, Acc = {accuracy:.2f}")
    fig.tight_layout()
    plt.savefig('results2/confusion_matrix.png')
    

## ---- skeletonize with medial axis ---- ##
from skimage.morphology import medial_axis
def skeletonize(img_data, index):
    '''skeletonize the image using the medial axis.'''
    # Preprocess the image
    img = get_standardized(img_data)

    # make binary
    img = img > THRESHOLD

    # find the medial axis
    skel, distance = medial_axis(img, return_distance=True)
    
    # plot the original and the skeleton
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(skel, cmap='gray')
    axs[1].set_title('Skeleton')
    plt.savefig(f'results2/skeleton_{index}.png')
    plt.show()
    return skel, distance

if __name__ == '__main__':
    if not os.path.exists('results2'):
        os.makedirs('results2')
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data() # run this once to save the data
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_train = y_train.astype(int)

    # find_cut_and_fit_line(zeros[0], show_plot=True)
    # find_cut_and_fit_line(ones[0], show_plot=True)

    # benchmark_all(X_train, y_train, num_times = 4029, cut_multiple=True)

    # X_test = np.load('data/X_test.npy', allow_pickle=True)
    # y_test = np.load('data/y_test.npy', allow_pickle=True)
    # y_test = y_test.astype(int)

    # predict_0_1(X_test, y_test, combined_chi2_path='results2/chi2_avg_sem_4029_True.json')
    # predict_all(X_test, y_test, combined_chi2_path='results2/chi2_avg_sem_4029_True.json')

    index = np.random.randint(0, len(X_train))
    skeletonize(X_train[index], index)
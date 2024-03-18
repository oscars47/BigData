# implement correction to bsh.py based on Gu's comments

import numpy as np
from mnist_2 import *
from vec_angles import *
from scipy.optimize import curve_fit
from oscars_toolbox.trabbit import trabbit
from functools import partial
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from scipy.ndimage import label

## fitting 0 and 1 ##
def fit_0_1(img, target=0):
    # fit curve to 0 and 1
    if target == 0:
        elp = EllipseModel()
        y, x = np.where(img == 1)
        # combine x and y
        x = np.vstack((x, y)).T
        elp.estimate(x)
        elp_params = elp.params

        return elp_params, elp

    # 1
    if target == 1:
        def curve_1(x, a, b):
            return x*a + b

        y1, x1 = np.where(img == 1)
        popt_1, pcov_1 = curve_fit(curve_1, x1, y1)

        return popt_1, x1, y1, curve_1

def create_means(X_train, y_train, show_plot=True):
    '''create means for each digit'''
    # split X_train into different sections based on y_train
    unique_targets = np.unique(y_train)
    
    X_train_split = {}
    for i in unique_targets:
        X_train_split[i] = X_train[y_train == i]

    # create means of the image for each digit
    means = {}
    for i in unique_targets:
        means[i] = np.mean([img for img in X_train_split[i]], axis=0)

    # perform skeletonization
    means_skels = {}
    for i in unique_targets:
        means_skels[i] = skeletonize(means[i].reshape(28, 28)).astype(int)

    if show_plot:
        # fit curves to 0 and 1
        elp_params = fit_0_1(means_skels[0], target=0)
        popt_1, x1, y1 = fit_0_1(means_skels[1], target=1)
    
        # plot them
        fig, axs = plt.subplots(2, len(unique_targets), figsize=(10, 10))
        for i in range(len(unique_targets)):
            axs[0][i].imshow(means[i].reshape(28, 28), cmap='gray')
            axs[0][i].set_title(f'Mean {i}')
            axs[1][i].imshow(means_skels[i], cmap='gray')
            axs[1][i].set_title(f'Skeletonized {i}')
            # add fits to skeletonized images
            if i == 0:
                xc, yc, a, b, theta = elp_params
                elp_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none', line28=2)

                axs[1][0].add_patch(elp_patch)
            elif i == 1:
                # 1
                def curve_1(x, a, b):
                    return x*a + b
                x_min = np.min(x1)
                x_max = np.max(x1)
                x = np.linspace(x_min, x_max, 100)
                y = curve_1(x, *popt_1)
                axs[1][i].plot(x, y, 'r', line28=2)
                # set bound as 0 and 28
                # axs[1][i].set_xlim([0, 28])
                # axs[1][i].set_ylim([0, 28])
        plt.savefig(f'results_skel/correct_means_{len(X_train)}.pdf')
        plt.show()

        # save each mean separately
        for i in unique_targets:
            np.save(f'results_skel/mean_{i}_{len(X_train_split[i])}.npy', means[i].reshape(28, 28))
            np.save(f'results_skel/mean_skel_{i}_{len(X_train_split[i])}.npy', means_skels[i])

        # fit all skeletons to 0 and 1
    
            X0_ls = X_train_split[0]
            X1_ls = X_train_split[1]
            # fit curve to 0 and 1
            elp_params_df = pd.DataFrame()
            one_angles = []
            for j in tqdm(X0_ls):
                j = j.reshape(28, 28)
                j = skeletonize(j).astype(int)
                elp_params = fit_0_1(j, target=0)
                # append to df
                elp_params_df = pd.concat([elp_params_df, pd.DataFrame(elp_params).T])
            
            for j in tqdm(X1_ls):
                j = j.reshape(28, 28)
                j = skeletonize(j).astype(int)
                popt_1, x1, y1 = fit_0_1(j, target=1)
                # get angle from slope
                angle = np.arctan(popt_1[0]) * 180/np.pi
                one_angles.append(angle)

            # add column names
            elp_params_df.columns = ['xc', 'yc', 'a', 'b', 'theta']
        
            # save elp_params_df
            elp_params_df.to_csv(f'results_skel/elp_params_df_{i}.csv')
            # save one_angles
            one_angles = np.array(one_angles)
            np.save(f'results_skel/one_angles_{i}.npy', one_angles)

            # plot all elp_params
            # xc, yc, a, b, theta
            params_label = ['$x_c$', '$y_c', '$a$', '$b$', '$\\theta$']
            fig, axs = plt.subplots(1, 5, figsize=(10, 5))
            for k in range(5):
                axs[k].hist(elp_params_df, bins=100)
                axs[k].set_title(f'{params_label[k]}')
            plt.savefig(f'results_skel/elp_params_hist_{params_label[i]}.pdf')

            # plot one_angles
            plt.figure(figsize=(10, 5))
            plt.hist(one_angles, bins=100)
            plt.savefig(f'results_skel/one_angles_hist_{i}.pdf')

    return means, means_skels

def replot_hist(el_params_df_path='results_skel/elp_params_df_0.csv', one_angles_path='results_skel/one_angles_0.npy'):
    elp_params_df = pd.read_csv(el_params_df_path)
    one_angles = np.load(one_angles_path)
    # drop first column
    elp_params_df = elp_params_df.drop(columns='Unnamed: 0')

    elp_params_df.columns = ['xc', 'yc', 'a', 'b', 'theta']

    params_label = ['$x_c$', '$y_c$', '$a$', '$b$', '$\\theta$']

    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
  
    for k in range(5):
        axs[k].hist(elp_params_df[elp_params_df.columns[k]], bins=100)
        axs[k].set_title(f'{params_label[k]}')
    plt.savefig(f'results_skel/elp_params_hist_{len(elp_params_df)}.pdf')

    # plot one_angles
    plt.figure(figsize=(10, 5))
    plt.hist(one_angles, bins=100)
    plt.savefig(f'results_skel/one_angles_hist_{len(one_angles)}.pdf')

## params for 0 and 1 relative to the mean ##
def get_params_rel_mean(X_train, Y_train, angle_spacing=10):
    means, means_skel = create_means(X_train, Y_train, show_plot=False)

    X1 = X_train[Y_train == 1]
    X0 = X_train[Y_train == 0]

    # do 1 first: angles and lengths
    angles = []
    lengths = []
    for x in X1:
        # get skeleton
        x = x.reshape(28, 28)
        x = skeletonize(x).astype(int)
        popt_1, x1, y1, curve_1 = fit_0_1(x, target=1)
        # get angle from slope
        angle = np.arctan(popt_1[0]) * 180/np.pi
        # get length
        # length = np.sqrt((x1[-1] - x1[0])**2 + (y1[-1] - y1[0])**2)
        # find min and max in x and y
        x_min, x_max = np.min(x1), np.max(x1)
        y_min, y_max = np.min(y1), np.max(y1)
        length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        # append
        angles.append(angle)
        lengths.append(length)

        # mark the points on the image
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(x, cmap='gray')
        # ax.plot(x_min, y_min, 'o', color='blue')
        # ax.plot(x_max, y_max, 'o', color='blue')
        # plt.savefig(f'results_skel/one_{len(X1)}_{angle}_{length}.pdf')
        # plt.show()

    # plot histogram of angles and lengths
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(angles, bins=100)
    axs[0].set_title('Angles')
    axs[1].hist(lengths, bins=100)
    axs[1].set_title('Lengths')
    plt.savefig(f'results_skel/one_angles_lengths_hist_{len(X1)}.pdf')

    # do 0
    # divide a circle into 10 degree segments and see how it maps onto the ellipse
    # first fit ellipse to mean
    steps = 360 // angle_spacing
    angles = np.linspace(0, 360, steps)
    def find_equal_points_elp(img):
        elp_params, elp = fit_0_1(img, target=0)
        # get angles equally spaced and find where they lie on the ellipse
        # get points on ellipse
        xy = elp.predict_xy(angles, params=elp_params)
        x, y = xy[:, 0], xy[:, 1]
        return x, y
    def find_di(x_i, y_i, x_targ, y_targ):
        return np.sqrt((x_i - x_targ)**2 + (y_i - y_targ)**2)
    def find_ri(x_i, y_i):
        return np.sqrt(x_i**2 + y_i**2)
    
    x_mean, y_mean = find_equal_points_elp(means_skel[0])
    x_0, y_0 = find_equal_points_elp(skeletonize(X0[0].reshape(28, 28)))
    x_1, y_1 = find_equal_points_elp(skeletonize(X1[0].reshape(28, 28)))

    m = find_ri(x_mean, y_mean)
    r_0 = find_ri(x_0, y_0)
    r_1 = find_ri(x_1, y_1)
    d_0 = find_di(x_0, y_0, x_mean, y_mean)
    d_1 = find_di(x_1, y_1, x_mean, y_mean)

    # log all distances
    d_ls = np.zeros((len(X0), steps))
    for i, x in enumerate(X0):
        x = x.reshape(28, 28)
        x = skeletonize(x).astype(int)
        x_i, y_i = find_equal_points_elp(x)
        d = find_di(x_i, y_i, x_mean, y_mean)
        d_ls[i] = d

    # plot m, r_o, r_1, d_0, d_1
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for ax[0], plot the original images
    axs[0].imshow(skeletonize(X0[0].reshape(28, 28)), cmap='gray', alpha=0.5, )
    axs[0].imshow(skeletonize(X1[0].reshape(28, 28)), cmap='gray', alpha=0.5, )
    axs[0].imshow(means_skel[0], cmap='gray', alpha=0.5)
    axs[0].scatter(x_mean, y_mean, color='red', label='Mean')
    axs[0].scatter(x_0, y_0, color='blue', label='0')
    axs[0].scatter(x_1, y_1, color='gold', label='1')
    axs[0].set_title('Original Images')
    axs[0].legend()

    # for axs[1], plot m, r_0, r_1, d_0, d_1
    axs[1].plot(m, label='Mean')
    axs[1].plot(r_0, label='$r_0$')
    axs[1].plot(r_1, label='$r_1$')
    axs[1].plot(d_0, label='$d_0$')
    axs[1].plot(d_1, label='$d_1$')
    axs[1].legend()
    axs[1].set_title('Distances')

    # make a histogram of d_ls at each angle
    # get colorwheel steps long
    colorwheel = plt.cm.viridis(np.linspace(0, 1, steps))
    for i in range(steps):
        d_i = d_ls[:, i]
        axs[2].hist(d_i, bins=100, alpha=0.5, color=colorwheel[i])        
    # plt.plot(m, label='Mean')

    axs[2].set_title('All Distances')

    plt.tight_layout()
    plt.savefig(f'results_skel/zero_distances_{len(X0)}.pdf')

def get_pca_params(X_train, Y_train):
    '''gets histogram of diff of angle of principal axis relative to mean, as well as 1st and 2nd singular values of PCA of each digit

    NOTE: only set up to plot for 0 and 1
    
    '''

    def get_pca_params_single(X):
        '''returns the rotation angle of the longer axis of PCA as well as the singular values'''
        X = X.reshape(28, 28)
        pca = PCA(n_components=2)
        pca.fit(X)
        # get singular values
        singular_values = pca.singular_values_
        # get rotation angle
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) * (180.0 / np.pi)
        return angle, singular_values
        
        
    # get the PCA parameters
    # get the PCA parameters
    X0 = X_train[Y_train == 0]
    X1 = X_train[Y_train == 1]
    # get PCA parameters
    params_0 = [get_pca_params_single(x0) for x0 in X0]
    params_1 = [get_pca_params_single(x1) for x1 in X1]

    angle_0 = np.array([i[0] for i in params_0])
    singular_vals_0 = np.array([i[1] for i in params_0])

    angle_1 = np.array([i[0] for i in params_1])
    singular_vals_1 = np.array([i[1] for i in params_1])
    
    # get mean angles and singular values
    mean_0 = np.mean(X0, axis=0)
    mean_1 = np.mean(X1, axis=0)

    mean_angle_0, mean_singular_vals_0 = get_pca_params_single(mean_0)
    mean_angle_1, mean_singular_vals_1 = get_pca_params_single(mean_1)

    angle_diff_0 = mean_angle_0 - angle_0
    angle_diff_1 = mean_angle_1 - angle_1

    # plot histograms, 2x3
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs[0][0].hist(angle_diff_0, bins=100)
    axs[0][0].set_title('Angle Difference 0')
    axs[0][1].hist(singular_vals_0[:, 0], bins=100)
    axs[0][1].vlines(mean_singular_vals_0[0], ymin=0, ymax=160, color='red', label='$\lambda_1$ of mean')
    axs[0][1].set_title('First Singular Values 0')
    axs[0][2].hist(singular_vals_0[:, 1], bins=100)
    axs[0][2].vlines(mean_singular_vals_0[1], ymin=0, ymax=120, color='red', label='$\lambda_2$ of mean')
    axs[0][2].set_title('Second Singular Values 0')

    axs[1][0].hist(angle_diff_1, bins=100)
    axs[1][0].set_title('Angle Difference 1')
    axs[1][1].hist(singular_vals_1[:, 0], bins=100)
    axs[1][1].vlines(mean_singular_vals_1[0], ymin=0, ymax=160, color='red', label='$\lambda_1$ of mean')
    axs[1][1].set_title('First Singular Values 1')
    axs[1][2].hist(singular_vals_1[:, 1], bins=100)
    axs[1][2].vlines(mean_singular_vals_1[1], ymin=0, ymax=160, color='red', label='$\lambda_2$ of mean')
    axs[1][2].set_title('Second Singular Values 1')
    axs[0][1].legend()
    axs[0][2].legend()
    axs[1][1].legend()
    axs[1][2].legend()
    plt.savefig(f'results_skel/pca_params_{len(X_train)}.pdf')

def dfs(matrix, start, end, path=[], visited=set()):
    '''depth first search for walking along the digit'''
    # Define the possible movements, including diagonals
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    # Base case: if start is the same as end, return the path including the end
    if start == end and len(path) > 1:
        return path + [end]
    
    # Mark the current point as visited
    visited.add(start)
    
    # Iterate through possible movements
    for move in movements:
        next_position = (start[0] + move[0], start[1] + move[1])
        
        # Check if the next position is valid and not visited
        if (0 <= next_position[0] < len(matrix)) and (0 <= next_position[1] < len(matrix[0])) and (next_position not in visited):
            # Additionally, ensure we're moving towards a brighter or equal point; assuming '1' is brighter than '0'
            if matrix[next_position[0]][next_position[1]] >= matrix[start[0]][start[1]]:
                # Perform DFS from the next valid position
                result = dfs(matrix, next_position, end, path + [start], visited)
                # If a result is found, return it
                if result is not None:
                    return result
                
    # If no path is found, implicitly return None to backtrack
    return None

## find points of max directions ##
def find_max_directions(img_data, threshold=120, show_plot=True, index=None, num_class=None):
    '''computes the number of directions it is possible to walk on '''
    img = img_data.reshape(28, 28)
    img = img > threshold
    img = medial_axis(img).astype(int)

    def count_directions(img):
        # go through each pixel and find the number of possible directions
        directions = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] == 1:
                    # get the number of directions, which is a square of 3x3 minus the center
                    directions[i, j] = np.sum(img[i-1:i+2, j-1:j+2]) - 1 # don't want to count the center pixel

        return directions
    
    def is_manifold(img):
        '''assumes there is only 1 manifold present'''
        directions = count_directions(img)
        # max of directions should be 2
        max_directions = np.max(directions)
        return max_directions <= 2
    
    def is_connected(img):
        '''assumes there is only 1 manifold present'''
        # is it possible to walk from one pixel back to itself without crossing the same pixel twice?
        # if it is, then it's connected
        # get the starting point
        y, x = np.where(img == 1)
        start = (y[0], x[0])
        end = start
     
        # use DFS to find if it's connected
        path = dfs(img, start, end)
        return path is not None
        
    directions = count_directions(img)

    # Find max directions
    # max_directions = np.max(directions)
    # print("Max directions:", max_directions)
    # max_directions_idx = np.where(directions == max_directions)
    max_directions_idx = np.where(directions > 2)

    max_directions_idx_y = max_directions_idx[0]
    max_directions_idx_x = max_directions_idx[1]

    if len(max_directions_idx_y) > 1:
        # Adjacency check function: within 1 pixel of each other
        def is_adjacent(idx1, idx2):
            return abs(idx1[0] - idx2[0]) <= 1 and abs(idx1[1] - idx2[1]) <= 1

        # Initialize groups of adjacent indices
        groups = []

        # Function to find the group an index belongs to
        def find_groups(idx, groups):
            belongs_to = []
            for i, group in enumerate(groups):
                if any(is_adjacent(idx, member) for member in group):
                    belongs_to.append(i)
            return belongs_to

        # Group adjacent indices
        for i in range(len(max_directions_idx_y)):
            current_idx = (max_directions_idx_y[i], max_directions_idx_x[i])
            idx_groups = find_groups(current_idx, groups)
            
            if idx_groups:
                # Merge groups if necessary
                if len(idx_groups) > 1:
                    new_group = set()
                    for g in idx_groups:
                        new_group = new_group.union(groups[g])
                    new_group.add(current_idx)
                    groups[idx_groups[0]] = new_group
                    for g in sorted(idx_groups[1:], reverse=True):
                        del groups[g]
                else:
                    groups[idx_groups[0]].add(current_idx)
            else:
                # Create new group if no existing group is found
                groups.append({current_idx})

        # make sure to add the non-adjacent
        for idx in zip(max_directions_idx_y, max_directions_idx_x):
            if not any(idx in group for group in groups):
                groups.append({idx})

        # Select rightmost-bottom from each group
        final_indices = [max(group, key=lambda x: (x[0], x[1])) for group in groups]

        print("Final indices (rightmost-bottom of each group):", final_indices)

    else:
        final_indices = [(max_directions_idx_y[0], max_directions_idx_x[0])]

    y, x = np.where(img == 1)
    center = (np.mean(y), np.mean(x))

    # remove the max connected points
    for i in final_indices:
        img[i[0], i[1]] = 0

    # separate out the manifolds, i.e. the separate connected components
    labeled_array, num_features = label(img, structure=np.ones((3, 3)))  

    # determine for each manifold if adding back the point makes it connected; if it does then it's closed
    closed = []
    # get separate manifolds
    for i in range(1, num_features+1):
        manifold = np.where(labeled_array == i, 1, 0)
        print(i)
        for j in final_indices:
            manifold = deepcopy(manifold)
            manifold[j[0], j[1]] = 1
            print(is_connected(manifold), is_manifold(manifold))
            if is_connected(manifold) and is_manifold(manifold):
                closed.append(True)
                break

    print('Closed:', closed)
       
    # plot img and heatmap
    if show_plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        cmap = axs[0].imshow(directions, cmap='hot')
        fig.colorbar(cmap, ax=axs[0])
        final_indices_x = [i[0] for i in final_indices]
        final_indices_y = [i[1] for i in final_indices]
        axs[0].scatter(final_indices_y, final_indices_x, color='blue')
        axs[0].scatter(center[1], center[0], color='green')
        axs[0].set_title('Heatmap of Directions')
        cmap2 = axs[1].imshow(labeled_array, cmap='nipy_spectral')
        fig.colorbar(cmap2, ax=axs[1])
        axs[1].set_title('Cut Skeletonized Image')

        if index is not None and num_class is not None:
            plt.savefig(f'results_skel/directions_{num_class}_{index}.pdf')
        else:
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'results_skel/directions_{time}.pdf')
        plt.show()

    # fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    # cmap = axs.imshow(directions, cmap='hot')
    # fig.colorbar(cmap, ax=axs)
    # final_indices_x = [i[0] for i in final_indices]
    # final_indices_y = [i[1] for i in final_indices]
    # axs.scatter(final_indices_y, final_indices_x, color='blue')
    # axs.scatter(center[1], center[0], color='green')
    # plt.savefig('results_skel/directions_tangent.pdf')
    # plt.show()


    # for i in final_indices:
    #     # Ensure context_window doesn't exceed image bounds
    #     x_min = max(i[1]-context_window, 0)
    #     x_max = min(i[1]+context_window, img.shape[1])
    #     y_min = max(i[0]-context_window, 0)
    #     y_max = min(i[0]+context_window, img.shape[0])
        
    #     # Get the 3x3 square around the point (with boundary checks)
    #     square = img[y_min:y_max+1, x_min:x_max+1]
        
    #     # Get the points
    #     y, x = np.where(square == 1)
        
    #     # Adjust x and y to the context of the whole image
    #     x = x + x_min
    #     y = y + y_min

    #     # Fit a line using polyfit, where x serves as the independent variable and y as the dependent
    #     slope, _ = np.polyfit(x, y, 1)

    #     # Plot the line
    #     # Define a range for x that covers the extent of the plot for visualization
    #     x_line = np.array([x_min, x_max])
    #     intercept = i[0] - slope * i[1]  # Adjust equation to use point as reference
    #     y_line = slope * x_line + intercept  # Adjust equation to use point as reference
    #     axs.plot(x_line, y_line, color='purple', linestyle='--')
    #     axs.vlines(i[1], ymin=y_min, ymax=y_max, color='purple', linestyle='dotted')

    #     # determine what quadrant relative to the center this point is
    #     # bottom right
    #     if i[0] > center[0] and i[1] > center[1]:
    #         print('bottom right')
    #         # remove the closest non-0 pixel to the right and bottom of the vertical line through the point and the tangent line
    #         # condition: x > i[1] and y > i[0] and y - slope*x - intercept > 0
    #         # get the points
    #         y, x = np.where(img == 1)
    #         # get the points that satisfy the condition
    #         condition = ((x > i[1]) | (x == i[1]))  & ((y > i[0]) |(y == i[0]))  & ((y - slope*x - intercept > 0) | (y - slope*x - intercept == 0)) 
    #         # condition = (x != i[1]) & (y != i[0]) & condition
    #         y, x = y[condition], x[condition]
    #         # get the distances
    #         # axs.scatter(x, y, color='gold')
    #         dists = np.sqrt((x - i[1])**2 + (y - i[0])**2)
    #         print(dists)
    #         # remove the closest point that satisfies the condition
    #         closest_point = (y[np.argmin(dists)], x[np.argmin(dists)])
    #         print('closest', closest_point)
    #         axs.scatter(closest_point[1], closest_point[0], color='magenta')
    #         img[closest_point[1], closest_point[0]] = 0
    #     # top right
    #     elif i[0] < center[0] and i[1] > center[1]:
    #         # remove the closest non-0 pixel to the right and top of the vertical line through the point and the tangent line
    #         # condition: x > i[1] and y < i[0] and y - slope*x - intercept < 0
    #         # get the points
    #         y, x = np.where(img == 1)
    #         # get the points that satisfy the condition
    #         condition = ((x > i[1]) | (x == i[1])) & ((y < i[0]) | (y == i[0])) & ((y - slope*x - intercept < 0) | (y - slope*x - intercept == 0))
    #         condition = (x != i[1]) & (y != i[0]) & condition
    #         y, x = y[condition], x[condition]
    #         # get the distances
    #         # axs.scatter(x, y, color='magenta')
    #         dists = np.sqrt((x - i[1])**2 + (y - i[0])**2)
    #         print(dists)
    #         # remove the closest point that satisfies the condition
    #         closest_point = (y[np.argmin(dists)], x[np.argmin(dists)])
    #         print('closest', closest_point)
    #         axs.scatter(closest_point[1], closest_point[0], color='magenta')
    #         img[closest_point[1], closest_point[0]] = 0
    #     # top left
    #     elif i[0] < center[0] and i[1] < center[1]:
    #         # remove the closest non-0 pixel to the left and top of the vertical line through the point and the tangent line
    #         # condition: x < i[1] and y < i[0] and y - slope*x - intercept > 0
    #         # get the points
    #         y, x = np.where(img == 1)
    #         # get the points that satisfy the condition
    #         condition = ((x < i[1]) | (x == i[1])) & ((y < i[0]) | (y == i[0])) & ((y - slope*x - intercept > 0) | (y - slope*x - intercept == 0))
    #         condition = (x != i[1]) & (y != i[0]) & condition
    #         y, x = y[condition], x[condition]
    #         # get the distances
    #         # axs.scatter(x, y, color='magenta')
    #         dists = np.sqrt((x - i[1])**2 + (y - i[0])**2)
    #         print(dists)
    #         # remove the closest point that satisfies the condition
    #         closest_point = (y[np.argmin(dists)], x[np.argmin(dists)])
    #         print('closest', closest_point)
    #         axs.scatter(closest_point[1], closest_point[0], color='magenta')
    #         img[closest_point[1], closest_point[0]] = 0

    #     # bottom left
    #     elif i[0] > center[0] and i[1] < center[1]:
    #         # remove the closest non-0 pixel to the left and bottom of the vertical line through the point and the tangent line
    #         # condition: x < i[1] and y > i[0] and y - slope*x - intercept < 0
    #         # get the points
    #         y, x = np.where(img == 1)
    #         # get the points that satisfy the condition
    #         condition = ((x < i[1]) | (x == i[1])) & ((y > i[0]) | (y == i[0])) & ((y - slope*x - intercept < 0) | (y - slope*x - intercept == 0))
    #         condition = (x != i[1]) & (y != i[0]) & condition
    #         y, x = y[condition], x[condition]
    #         # get the distances
    #         # axs.scatter(x, y, color='magenta')
    #         dists = np.sqrt((x - i[1])**2 + (y - i[0])**2)
    #         print(dists)
    #         # remove the closest point that satisfies the condition
    #         closest_point = (y[np.argmin(dists)], x[np.argmin(dists)])
    #         print('closest', closest_point)
    #         axs.scatter(closest_point[1], closest_point[0], color='magenta')
    #         img[closest_point[1], closest_point[0]] = 0

        # Draw box around fit region
        # axs.plot([x_min, x_max, x_max, x_min, x_min], 
        #         [y_min, y_min, y_max, y_max, y_min], color='purple')
                       
    # # replot img and old img
    # if show_plot:
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     axs[0].imshow(img_old, cmap='gray')
    #     axs[0].set_title('Original Image')
    #     axs[1].imshow(img, cmap='gray')
    #     axs[1].set_title('Cut Image')
    #     plt.savefig('results_skel/cut_img.pdf')
    #     plt.show()

    return img

if __name__ == '__main__':
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

    index = 0
    find_max_directions(eights[index], index=index, num_class=8)



    # zero_one_mini = np.concatenate((zeros[:100], ones[:100]), axis=0)
    # zero_one_y_mini = np.array([0]*100 + [1]*100)
    # zero_one_total = np.concatenate((zeros, ones), axis=0)
    # zero_one_y_total = np.array([0]*len(zeros) + [1]*len(ones))
    # create_means(zero_one_total, zero_one_y_total)
    # nines_y = np.array([9]*len(nines))
    # cut_walk_main(nines, show_start_points=True)

    # create_means(nines, nines_y)
    # average the means
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # nine_means = np.mean(nines, axis=0).reshape(28, 28)
    # axs[0].imshow(nine_means, cmap='gray')
    # # maximal pixel brightness
    # y, x = np.where(nine_means > 0)
    # y_max, x_max = y[np.argmax(nine_means[y, x])], x[np.argmax(nine_means[y, x])]
    # axs[0].scatter(x_max, y_max)
    # axs[1].scatter(x_max, y_max)
    # # get skeleton
    # nine_means = nine_means > 90
    # nine_skel = medial_axis(nine_means).astype(int)
    # axs[1].imshow(nine_skel, cmap='gray')
    # plt.savefig(f'results_skel/nine_means_{len(nines)}.pdf')
    # plt.show()

    # replot_hist()


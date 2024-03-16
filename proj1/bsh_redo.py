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
                elp_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none', linewidth=2)

                axs[1][0].add_patch(elp_patch)
            elif i == 1:
                # 1
                def curve_1(x, a, b):
                    return x*a + b
                x_min = np.min(x1)
                x_max = np.max(x1)
                x = np.linspace(x_min, x_max, 100)
                y = curve_1(x, *popt_1)
                axs[1][i].plot(x, y, 'r', linewidth=2)
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

## walking along digit ##
def cut_walk(img, brightest_point, threshold = 90, show_start_points=True):
    '''cut the digit by walking along the digit. currently only works for 9'''
    # get skeleton
    img = img.reshape(28, 28)
    img = img > threshold
    img = medial_axis(img).astype(int)

    # get bounding box
    y, x = np.where(img == 1)

    # where a non-zero pixel touches the edge of the bounding box, start 
    start_points = []
    # have to consider 4 sides
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # top
    for i in range(x_min, x_max):
        if img[y_min, i] == 1:
            start_points.append((i, y_min))
    # bottom
    for i in range(x_min, x_max):
        if img[y_max, i] == 1:
            start_points.append((i, y_max))

    # left
    for i in range(y_min, y_max):
        if img[i, x_min] == 1:
            start_points.append((x_min, i))

    # right
    for i in range(y_min, y_max):
        if img[i, x_max] == 1:
            start_points.append((x_max, i))

    # reassign brightest_point to closest non-zero pixel
    x, y = np.where(img == 1)
    # get closest point
    dists = np.sqrt((x - brightest_point[0])**2 + (y - brightest_point[1])**2)
    closest_point = (x[np.argmin(dists)], y[np.argmin(dists)])
        

    if show_start_points:
        # plot image and start points
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(img, cmap='gray')
        # show bounding box
        ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='red')
        for i in start_points:
            ax.scatter(i[0], i[1], color='red')
        ax.scatter(brightest_point[1], brightest_point[0], color='blue')
        ax.scatter(closest_point[1], closest_point[0], color='green')
        plt.savefig('results_skel/start_points.pdf')
        plt.show()

    # now walk along the digit and cut at intersection point of all paths
    # walk towards the brightest point using DFS
    def dfs(matrix, start, end, path=[], visited=set()):
        # Define the possible movements, including diagonals
        movements = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # Base case: if start is the same as end, return the path including the end
        if start == end:
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
    
    # swap x and y in start_points
    start_points = [(j, i) for i, j in start_points]
    print(len(start_points))
        
    paths = []
    for point in start_points:
        path = dfs(img, point, closest_point)
        if path is not None:
            paths.append(path)

    # plot the paths
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap='gray')
    for i, path in enumerate(paths):
        path = np.array(path)
        ax.scatter(start_points[i][1], start_points[i][0])
        ax.plot(path[:, 1], path[:, 0])
    ax.scatter(closest_point[1], closest_point[0], color='green')
    plt.savefig('results_skel/paths.pdf')
    plt.show()
    # find intersection of all paths
        

def cut_walk_main(imgs, threshold=90, show_start_points=True):

    means = np.mean(imgs, axis=0).reshape(28, 28)
    # maximal pixel brightness
    y, x = np.where(means > 0)
    y_max, x_max = y[np.argmax(means[y, x])], x[np.argmax(means[y, x])]
    brightest_point = (y_max, x_max)
    # get skeleton
    means = means > 90
    skel = medial_axis(means).astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(skel, cmap='gray')
    ax.imshow(means, cmap='gray', alpha=0.5)
    ax.scatter(x_max, y_max)
    plt.savefig('results_skel/brightest_point.pdf')

    for img in imgs:
        cut_walk(img, brightest_point, show_start_points=show_start_points, threshold=threshold)



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

    # get_params_rel_mean(X_train,  y_train)

    # zero_one_mini = np.concatenate((zeros[:100], ones[:100]), axis=0)
    # zero_one_y_mini = np.array([0]*100 + [1]*100)
    # zero_one_total = np.concatenate((zeros, ones), axis=0)
    # zero_one_y_total = np.array([0]*len(zeros) + [1]*len(ones))
    # create_means(zero_one_total, zero_one_y_total)
    nines_y = np.array([9]*len(nines))
    cut_walk_main(nines, show_start_points=True)

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


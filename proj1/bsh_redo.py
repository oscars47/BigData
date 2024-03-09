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

def fit_0_1(img, target=0):
    # fit curve to 0 and 1
    if target == 0:
        elp = EllipseModel()
        y, x = np.where(img == 1)
        # combine x and y
        x = np.vstack((x, y)).T
        elp.estimate(x)
        elp_params = elp.params

        return elp_params

    # 1
    if target == 1:
        def curve_1(x, a, b):
            return x*a + b

        y1, x1 = np.where(img == 1)
        popt_1, pcov_1 = curve_fit(curve_1, x1, y1)

        return popt_1, x1, y1

    

def create_means(X_train, y_train):
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


    return means

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

    # zero_one_mini = np.concatenate((zeros[:100], ones[:100]), axis=0)
    # zero_one_y_mini = np.array([0]*100 + [1]*100)
    # zero_one_total = np.concatenate((zeros, ones), axis=0)
    # zero_one_y_total = np.array([0]*len(zeros) + [1]*len(ones))
    # create_means(zero_one_total, zero_one_y_total)
    nines_y = np.array([9]*len(nines))

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

    replot_hist()


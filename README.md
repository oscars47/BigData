Code for Math 179 (Big Data at HMC) with Prof. Gu.

# Project 1: Predicting MNIST digits using direct algorithm

## 2/20/24
* what gu wants:
    Please cut each digit to the top and bottom pieces. Then click a piece is linear you don’t need to transfer to polor coordinates. 

    To decide whether that piece is linear or not, just use linear regression 

    The goal is to represent a digit only by a few curve pieces that the computer can understand each piece which can be represented by a mathematical function. 

    Key is to make a thick stroke to a very thin curve which can be written as a function 

* what we're doing:
    considering only 0 and 1 digits for now,

    creating parameter of where to cut--based on number of points in that slice? for now just make it a hyperparam.
    
    fit line through the points in each section and plot residuals.

* added ```find_cut_fit_line``` and ```find_sig_max_density_x``` to plot_polar.py to perform the cutting based on finding maximum density slicing along x axis (only selected if max density - min density > threshold, otherwise use x center), and then fit to a line in each section and compute chi2red. we find chi2red dist is quite different whether digit is 0 (basically 0) and 1 (between 0.5-1) --- we wrote ```benchmark_0_1``` to test this.




## 2/13/24
* added mnist_2.py, which standardizes images by finding a bounding box, rescaling, performing PCA to rotate the images
* in plot_polar.py, applies standardization from mnist_2.py and then converts all the positions to polar coordinates.

## 2/12/24
* added mnist_idea.py, which creates networks by dividing the images into grids and then assinging nodes and edges based on if neighboring nodes were adjacent in terms of their bounding boxes.
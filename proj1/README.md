# Project 1: Predicting MNIST digits using direct algorithm
video presentation 1: https://youtu.be/z8MzZwYQ8K0

## 3/27/24
* start 2pm-2:37pm, 4:45-5:40pm, 8:20pm-11pm. 1:55am - 3:30am. 4:10 am - 
* goal: understand the components of the cuts
    - removed the automatic add back connectivity feature
    - updated alpha = 3e-1, min_threshold=160
    - how to deal with >1 max connected removed point..?
    - what about this: fit spline through start, middle, and end points of each manifold in order to smooth the digits. recombine the pieces. in ```bsh_redo.py```
        - use scipy.interpolate.interp1d
        - stitch togehter 
        - now we can analyze the curvature as we traverse the digit
        - need to account for case where we dont need to cut anything like a 1 or 0
        - for cut pieces:
            - check if manifold is open or closed. fit spline to evenly sampled points around the manifold. need parametric spline (splev, splprep). no problem with this is how the y and x are organized is not the same as 
            - using the same appraoch we had for angles earlier which samples as the point walks for angles, except using all pointsd with a smoothing factor of c = 2 for quadratic.
        - now need to connect the splines together and measure curvature
            - results for 3 and 9 suggest that there is a common pattern
            - use curvature results for mean. need to set correct thresholds. used estimate and then manually edited the .text files for the images
    - need to fix the target digits manually and rerun and save curvature plots. *plan*: use curvature to compare 0 vs 1 vs 9. then compare all digits
        - fixed target digits
        - thought for later: we might need to have different example of 7 with a bar
        - using dynamic time warping to quantify similarity. the ```dtw-python``` package does slightly better (about .015) than the custom.
            0.413 on first 1000 train for 0, 1, 9
            0.446 on first 1000 test for 0, 1, 9

            0.705 on first 1000 train for 0, 1
            0.68 on first 1000 test for 0, 1 

            for all data:
                train
                    0, 1: 0.7015552753257671
                    0, 1, 9: 0.4440639269406393
                    all: 0.23573660714285713

            adding back non-manifold increases connectivity and hence accuracy:

            on first 1000 digits, s=20:
                train: 
                    0,1: 0.853
                    0,1,9: 0.597
                test:
                    0,1: 0.848
                    0,1,9: 0.57

        - what about using maximum cross-correlation coefficient?
            - like 0 accuracy...no

## 3/26/24
* start 1:50pm, end 2pm. 4:45pm - 6:04pm. 7:19pm-9:33pm
* thresholding
    - from histograms, appears that 150 is the min, so using that
* doing angles
    - need to keep track of points we remove?
    - added ```get_angles```, but not interting all the way through the omage for some reason
* making presentation for meeting

## 3/25/24
* start 1:20pm, end 2:40pm
* strategizing about how to classify the digits. plan is to first fix the thresholding by considering the total number of squares > 50, then cut. then apply the angle method to every point in the manifolds stitched together.
    - added ```determine_threshold```, which replaced the previous function of same name. runs very fast, so there must've been some other problem when i tried this same idea before.

    thinking ```threshold = 240 - |img_mean/200|```

    - found cv2 has an adaptive thresholding function that works on specific regions of the image. doesn't work well to preserve overall important features

## 3/19/24
* start 4:35pm, end 7pm, including meeting. 8:30pm start, 8:40pm. 
* first getting mean, std of the cut pieces for each digit
    - ```find_max_directions()``` in ```bsh_redo.py```
    - removed lone points (connectivity <2)
* then will fit line to mean of angles for each digit, use this to predict
    - fixed issue where didn't include exactly ```num_keep```
    - added ```predict_angles``` using ```angles.csv```.
    - 29.73% accurate

* get all labeled graph
    -  use skeleton

## 3/18/24
* start 3pm, end 7pm (2 hours dasion lit review)
* including graph theory approach for presentation; back to ```mnist_idea.py```. using igraph instead of networkx
* vector idea: place points in places of high connectivity

## 3/17/24
* presentation: 2:30pm - 3:30pm. 7:50pm start , 8:58pm finish. // 10 pm start, 12:23 am finish.
* just removing blue dots to create manifolds, then add them back once we know which are the separate pieces. not sure how interesting the question of "closed" is since you can have images that are broken up into multiple manifolds and won't ever be closed
    - realized have to consider not just max connectivity, but where connectivity > 2. this will only be 3
    - added separate functions within find_max_directions() for count_directions() and is_manifold()
* need to fix the region assignment for max connectivity.  only using the bottom right leads to not sometimes not having manifold
* trying to determine adaptive thresholding

## 3/16/24
* start 10pm, end 12am. 3:10 am start, 3:25 am finish.
* n.b. maybe "max connectivity" is better term than "max direction"
* continuing cutting: 
    - calculate tangent line at the point. also draw vertical line through the point. this creates a grid. if point is below and to the right of the center, then remove the closest non0 pixel in the bottom right of this grid, and similar for other regions.

## 3/15/24
* start at 12:20pm, lunch at 1:25 pm. 2:50 pm, end at 4pm. start at 10pm, end 12:am
    - had meeting 3-4.
* working on cut_walk(). weird error with start points not being populated correctly. also, what to do about multiple possible paths? maybe combine neigboring ones that within 1 pixel of each other to 1, then make sure you don't traverse where others have gone?
* after meeting, realized need to 
    1. fix the 0 processing: PCA for 0 and 1, in bsh_redo.py
    2. for cutting, find the max number of directions available and cut there. 
* implemented 1 in bsh_redo.py.
* started cutting. found point of max directions; grouped adjacent max direction points such that we always choose the bottom right


## 3/10/24
* start 8:40 pm, break 9:30pm. start 9:45 pm, 11:06 pm break
* final 0/1 processing
    - for 0: log difference d_i
    - for 1: log length of 1 line.  
        - don't think we need to implement proposed idea of forcing all 1 lines to go through the center bright point since that's just translation to a new center. fit the line, then fit it across the bounding box. log this length
* added ```cut_walk()``` for 9s

## 3/5/24
* 4:30 start, 7:19 end (w meeting at 5:30)
* changing target images. made file bsh_redo.py
    - for 0, take mean of all and then skeleton, then fit curve.
    - for 1, just fit a line to the skeleton and then measure angle.

## 3/4/24
* 4:00pm start, 5:32 finish 
* made plots of everything. making presentation

## 3/3/24
* 6:56 pm start, break at 8:52pm
* 9:39pm start, break at 11:49pm
* 4:25 am start,  5:41 am finish
* what we want to do:
    - fix the bounding box
    - for 0 and 1, overplot all of the skeletons
    - need function to "unskeletonize"
        - plot distribution about the skeletons for 0 and 1
    - find the beta, s, h params for each using procustes
    
* what we did: 
    - fixed ```get_standardized()```: first apply ```rotate_pca```, then apply threshold, then ```find_bounding_box```. this leads to best results for skeletons. added seprate function to adjust the aspect ratio when scaling up from base bounding box or to preserve aspect ratio and thus add padding
    - writing new file ```bsh.py```.
        - here whole point is to not standardize so we figure out how much to scale/rotate
        - added ```create_target()``` function to generate target images for each class. 2-9 are pretty straight so we will need to make more "smooth"
        - finding procustes distance to target. procustes does not include shearing. interestingly disparity
            - what if we use disparity directly to do classification? on first 100 of train images, got 0.25 accuracy.
            - ```scipy.spatial.procustes``` doesn't return the optimal scaling etc params, so writing custom function ```find_bsh``` to do this
                - get weird artifacts
            - compare based on PCA to get angle? check difference in angles. 
                - got example of pca comparison and angles
                - got data for all so can plot! make slides of this: ```comp_bsh_pca()```. quite fast, about 70 it/s.
        - plotted first mean of 100 0s and 1s: gives a sense of the spread in skeletons
    - still to do: 
        - (tomorrow) make plots of bsh comparison of 0 vs 1 data
        - (tomorrow) create new digits
        - (later) cut other digit types and decompose into parts (clarify w Gu)



## 2/27/24
* worked on presentation, also added plot of angles; added path only connectivity

## 2/26/24
* started lit review at 8:30pm.
* trying out larry's idea to train NN based on training on the angles of vectors: ```vec_angles.py```
* worked til 11:42. built recursive alg to explore the skeleton from a starting point and then select only 1/5 of the points to deal with noise.
* worked 3:30am - 4:30am. considered connecting points based on both minimum distance and minimum cost to travel along the non0 pixels. improvement from just naively connecting the selected pixels but still difficult to deal with disconnected sections in the original which translate to disconnections in the skeleton and thus to the simplified skeleton
* fixed number of red dots per skeleton so this way we can compute the angles which are a fixed number and then feed that into NN
* We can order the angles based on x position since the start points won’t be same

## 2/21/24
* testing out idea to run cut and fit on all digits to see how the digit maps to distribution of chi2 maps
** massively increased computational speed
** ran total num 1000 and 10000
** result illustrates clear relationship between chi2red of L,R and digit identity, which suggests we can determine digit placement by these statistics
* realized sometimes we cut at the very last index, so because of that considering combined L and R chi2
* strategy of closest difference: 0.2746 accuracy... see confusion matrix
* if we only compared 0 to 1, got accuracy of 0.93235
* what if we used multiple cuts? say take initial 2, then split 2 into 2 each for total of 4: 0 vs 1 accuracy increases to 0.988, but with other digits decreases to 0.272
* added code for medial axis-- will work on this for later:
    That is to use the algorithm to connect nearest point from the current point to the next. Then find angles of two consecutive vectors of P1 to P2, and P2 to P3, if the angle is too large above certain threshold then stop and cut. Thanks 

    We will try to use this method to iteratively achieve better results and make each piece to have a function representation. Then study its deformations and define equivalent classes.


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
** see https://docs.google.com/presentation/d/1qfD0HpdRbFuNRSLlWijFRz0Avbvnl0FUcPCgRj7Q0yU/edit#slide=id.p for pres link




## 2/13/24
* added mnist_2.py, which standardizes images by finding a bounding box, rescaling, performing PCA to rotate the images
* in plot_polar.py, applies standardization from mnist_2.py and then converts all the positions to polar coordinates.

## 2/12/24
* added mnist_idea.py, which creates networks by dividing the images into grids and then assinging nodes and edges based on if neighboring nodes were adjacent in terms of their bounding boxes.
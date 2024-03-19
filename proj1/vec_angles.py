import os
import numpy as np
import matplotlib.pyplot as plt
from mnist_2 import *
from skimage.morphology import medial_axis, skeletonize
from collections import deque


## ---- skeletonize with medial axis ---- ##
def skeletonize(img_data, show=False, standardize=True):
    '''skeletonize the image using the medial axis and return a vector of angles.'''
    # Preprocess the image
    if standardize:
        img = get_standardized(img_data)
    else:
        img = img_data.reshape(28, 28)
    # find the medial axis
    skel, _ = medial_axis(img, return_distance=True)
    skel = skel > 0

    if show:
        img_orig = img_data.reshape(28, 28)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(img_orig, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(img, cmap='gray')
        axs[1].set_title('Standardized Image')
        axs[2].imshow(skel, cmap='gray')
        axs[2].set_title('Skeleton')
        plt.show()
    return skel

## ---- finding angles ---- ##
def get_angle(nx, ny, x, y):
    dx, dy = nx - x, ny - y
    return np.arctan2(dy, dx) 

def thin_skeleton(skel, thinning_percentage=80):
    total_pixels = np.count_nonzero(skel)
    target_removal = int(total_pixels * (thinning_percentage / 100.0))
    removed = 0

    # Function to find edge pixels
    def find_edge_pixels(skel):
        edge_pixels = []
        for x in range(1, skel.shape[0] - 1):
            for y in range(1, skel.shape[1] - 1):
                if skel[x, y] and np.sum(skel[x-1:x+2, y-1:y+2]) < 4:  # Less connected, hence an edge
                    edge_pixels.append((x, y))
        return edge_pixels

    while removed < target_removal:
        edge_pixels = find_edge_pixels(skel)
        if not edge_pixels:
            break  # Stop if no more edge pixels can be safely removed
        for x, y in edge_pixels:
            if removed >= target_removal:
                break
            skel[x, y] = 0  # Remove pixel
            removed += 1

    return skel

def dfs(skel, x, y, visited):
    """
    Depth-First Search to traverse non-zero pixels in the skeleton.
    
    skel: 2D numpy array representing the skeletonized image.
    x, y: Current coordinates to explore.
    visited: Set of already visited coordinates (x, y).
    """
    # Define the four possible movements (up, down, left, right, diagonals)
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    # Mark the current position as visited
    # visited.add((x, y))
    visited.append((x, y))
    
    for dx, dy in movements:
        nx, ny = x + dx, y + dy
        
        # Check if the new position is within bounds and is a non-zero pixel that hasn't been visited yet
        if 0 <= nx < skel.shape[0] and 0 <= ny < skel.shape[1] and skel[nx, ny] != 0 and (nx, ny) not in visited:
            dfs(skel, nx, ny, visited)

def start_traversal(skel, start_x, start_y, num_keep):
    """
    Start traversing the skeleton from a specific starting point.
    
    skel: 2D numpy array of the skeleton.
    start_x, start_y: Starting coordinates.
    """
    visited = []
    dfs(skel, start_x, start_y, visited)
    # visited = list(visited)
    print(f'Visited {len(visited)} pixels')
    # select out only 1 in 5 coordinates
    # perc to keep = total pixels / num_keep
    frac = len(visited) // num_keep
    visited = visited[::frac]
    return visited  # Returning visited for visualization or further processing
    
def bfs_path_length(skel, start, end):
    """Return the number of steps in the shortest path between start and end points through non-zero pixels using BFS."""
    if start == end:  # If start and end points are the same
        return 0
    
    queue = deque([(start, 0)])  # Queue holds tuples of (point, distance)
    visited = set([start])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]  # 4-connected neighbors
    
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == end:
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < skel.shape[0] and 0 <= ny < skel.shape[1] and skel[nx, ny] != 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return -1  # Return -1 if no path exists

def find_nearest_with_path(skel, start, points, dist_opt=0):
    """Find the nearest point to 'start' in 'points' for which a path exists.

    Params:
    skel: 2D numpy array representing the skeleton.
    start: Tuple (x, y) representing the starting point.
    points: 2D numpy array of points to search for the nearest one.
    dist_opt: 0 for euclidean distance, 1 for path length only, 2 for both.
    
    
    """
    min_distance = np.inf
    min_cost = np.inf
    nearest_point = None
    for point in points:
        if dist_opt==2:
            path_cost= bfs_path_length(skel, tuple(start), tuple(point))
            print(f'Path cost: {path_cost}')
            if path_cost != -1:
                distance = np.linalg.norm(start - point)
                if distance < min_distance and path_cost < min_cost:
                    min_distance = distance
                    nearest_point = point
                    min_cost = path_cost
        elif dist_opt==1:
            distance = np.linalg.norm(start - point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
        else:
            path_cost= bfs_path_length(skel, tuple(start), tuple(point))
            if path_cost != -1 and path_cost < min_cost:
                min_cost = path_cost
                nearest_point = point
    return nearest_point

def connect_points_with_path(skel, points, dist_opt=0):
    """Connect points considering the nearest neighbor with a valid path."""
    if len(points) == 0:
        return []
    
    connected_points = [points[0]]
    pairs=[]
    points_list = points.tolist()
    points_list.pop(0)
    
    while points_list:
        current_point = connected_points[-1]
        # if np.all(current_point == points[0]):
        nearest_point = find_nearest_with_path(skel, current_point, np.array(points_list), dist_opt=dist_opt)
        # else:
        #     nearest_point = find_nearest_with_path(skel, current_point, np.array(points_list+points[0]))
        if nearest_point is not None:
            connected_points.append(nearest_point)
            points_list.remove(nearest_point.tolist())
            pairs.append((current_point, nearest_point))
        else:
            break  # No valid path to any remaining points
    
    return pairs
        
def calculate_angles(skel, const=10):
    '''calculate the angles between the pixels in the skeleton.'''

    # get a non-zero pixel
    x, y = np.nonzero(skel)
    # start with upper left corner, which is min in x and y
    x_min, y_min = np.min(x), np.max(y)
    # find the closest pixel to the upper left corner
    min_distance = np.inf
    x_s, y_s = 0, 0
    for i in range(len(x)):
        distance = np.sqrt((x[i] - x_min)**2 + (y[i] - y_min)**2)
        if distance < min_distance:
            min_distance = distance
            x_s, y_s = x[i], y[i]
    print(f'Starting at: {x_s}, {y_s}')

    visited = start_traversal(skel, x_s, y_s, const)

    # connect the points
    pairs_path_only = connect_points_with_path(skel, np.array(visited), dist_opt=0)
    pairs_distance_only = connect_points_with_path(skel, np.array(visited), dist_opt=1)
    pairs_both = connect_points_with_path(skel, np.array(visited), dist_opt=2)

    # calculate the angles between the path_only
    angles = []
    for (x1, y1), (x2, y2) in pairs_path_only:
        dx, dy = x2 - x1, y2 - y1
        angle = np.arctan2(dy, dx) 
        angles.append(angle)
                 
    return angles, visited, pairs_path_only, pairs_distance_only, pairs_both

def get_angles(img_data, index, show=False):
    '''get the angles from the skeletonized image.'''
    # Preprocess the image
    img = img_data.reshape(28, 28)
    y, x =  np.where(img > 0)
    threshold_y = np.mean(y)
    threshold_x = np.mean(x)
    mean_bright = np.mean([threshold_y, threshold_x])
    threshold = 240 - np.abs(mean_bright / 14)*52
    img = img > threshold
    skel = medial_axis(img)
    # calculate the angles
    angles, visited, pairs_path_only, pairs_distance_only, pairs_both = calculate_angles(skel)
    
    # plot the original and the skeleton
    if show:
        # img = get_standardized(img_data)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(skel, cmap='gray')
        axs[1].set_title('Skeleton')
        axs[1].scatter([y for x, y in visited], [x for x, y in visited], c='r')

        for (x1, y1), (x2, y2) in pairs_path_only[:-1]:
            axs[1].plot([y1, y2], [x1, x2], 'b', linewidth=2, alpha=0.7)
        # add label the last point to denote this is with pairs_path
        (x1, y1), (x2, y2) = pairs_path_only[-1]
        axs[1].plot([y1, y2], [x1, x2], 'b', linewidth=2, alpha=0.7, label='Path')
        # add a label to the start point
        initial_point = pairs_path_only[0][0]
        axs[1].text(initial_point[1], initial_point[0], 'Start', fontsize=12, color='b', alpha=1)
        # add a label to the last point
        final_point = pairs_path_only[-1][-1]
        axs[1].text(final_point[1], final_point[0], 'End', fontsize=12, color='b', alpha=1)

        for (x1, y1), (x2, y2) in pairs_distance_only:
            axs[1].plot([y1, y2], [x1, x2], 'g', linewidth=2, alpha=0.7, linestyle='--')

        # add label the last point to denote this is with pairs_distance
        (x1, y1), (x2, y2) = pairs_distance_only[-1]
        axs[1].plot([y1, y2], [x1, x2], 'g', linewidth=2, alpha=0.7, label='Distance', linestyle='--')
        initial_point = pairs_distance_only[0][0]
        axs[1].text(initial_point[1], initial_point[0], 'Start', fontsize=12, color='g', alpha=1)
        # add a label to the last point
        final_point = pairs_distance_only[-1][-1]
        axs[1].text(final_point[1], final_point[0], 'End', fontsize=12, color='g', alpha=1)


        for (x1, y1), (x2, y2) in pairs_both:
            axs[1].plot([y1, y2], [x1, x2], 'y', linewidth=2, alpha=0.7, linestyle='dashdot')

        # add label the last point to denote this is with pairs_both
        (x1, y1), (x2, y2) = pairs_both[-1]
        axs[1].plot([y1, y2], [x1, x2], 'y', linewidth=2, alpha=0.7, label='Both', linestyle='dashdot')

        initial_point = pairs_both[0][0]
        axs[1].text(initial_point[1], initial_point[0], 'Start', fontsize=12, color='y', alpha=1)
        # add a label to the last point
        final_point = pairs_both[-1][-1]
        axs[1].text(final_point[1], final_point[0], 'End', fontsize=12, color='y', alpha=1)
        axs[1].legend()

        # plot the angles
        # sort angles based on the x coordinate
        # pairs_path_only.sort(key=lambda x: x[0][0])
        # angles = np.array(angles)
        # angles = angles[pairs_path_only[0][0][0]:pairs_path_only[-1][0][0]]
        axs[2].plot(angles, 'b')
        axs[2].set_title('Angles')

        plt.savefig(f'results2/skeleton_{index}.pdf')
        plt.show()

        # count numbers of non-zero pixels
        n_pixels = np.count_nonzero(skel)
        print(f'Number of pixels in the skeleton: {n_pixels}')

        print(f'Num Angles: {len(angles)}')

        return skel, angles
    else:
        return angles

if __name__ == '__main__':
    # Load the data
    if not os.path.exists('results_skel'):
        os.makedirs('results_skel')
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data() # run this once to save the data
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_train = y_train.astype(int)

    index = np.random.randint(0, len(X_train))
    get_angles(X_train[index], index, show=True)


#    for x in range(skel.shape[0]):
#         for y in range(skel.shape[1]):
#             if skel[x, y]:  # If there's a pixel
#                 already_visited.append((x, y))
#                 neighbors = find_neighbors(x, y, skel)
#                 for (nx, ny) in neighbors:
#                     if (nx, ny) not in already_visited:
#                         dx, dy = nx - x, ny - y
#                         angle = np.arctan2(dy, dx) 
#                         angles.append(angle)

#                     # display the vector and angle
#                     if debug:
#                         plt.imshow(skel, cmap='gray')
#                         plt.plot([y, ny], [x, nx], 'r', linewidth=0.5, alpha=0.7)
#                         plt.text((y+ny)/2, (x+nx)/2, f'{angle:.2f}', fontsize=5, color='r')
#                         plt.show()
                    
#     return angles
    
    # def find_neighbors(x, y, skel):
    # '''find the neighbors of a pixel in the skeleton.'''
    # # nieghbor is defined as within 1 pixel of the current pixel
    # neighbors = []
    # for dx in range(-1, 2):
    #     for dy in range(-1, 2):
    #         nx, ny = x + dx, y + dy
    #         if (dx != 0 or dy != 0) and 0 <= nx < skel.shape[0] and 0 <= ny < skel.shape[1] and skel[nx, ny]:
    #             neighbors.append((nx, ny))

    # return neighbors
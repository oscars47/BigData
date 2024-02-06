# implement topological analysis on mnist dataset

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx

def prep_img(img_data, index, threshold=240, block_size=2):
    '''Converts image into network based on thresholding and block size'''
    img = img_data.reshape(28, 28)
    # Thresholding
    threshold = 128  # Example threshold
    binary_image = img > threshold

    # Initialize the graph
    G = nx.Graph()

    # Subdivide the binary image into blocks and assign nodes
    nodes = []

    # iterate over the image in zig zag pattern, starting at bottom left
    # prepare iterable for zig zag pattern
    node_id = 0
    # Modify the iteration to go in a zigzag pattern starting from the bottom left
    for i in range(img.shape[0] - block_size, -1, -block_size):
        # Determine the direction of the j iteration based on the row
        if ((img.shape[0] - i) // block_size) % 2 == 0:
            # Even row number (from the bottom), go left to right
            j_values = range(0, img.shape[1], block_size)
        else:
            # Odd row number (from the bottom), go right to left
            j_values = range(img.shape[1] - block_size, -1, -block_size)
        
        for j in j_values:
            block = binary_image[i:i+block_size, j:j+block_size]
            # Calculate weighted average position within the block
            if np.any(block):
                ys, xs = np.nonzero(block)  # Get indices of non-zero (true) elements
                ys = ys + i  # Adjust indices based on the block's position
                xs = xs + j
                weighted_y = np.mean(ys)
                weighted_x = np.mean(xs)
                
                # Add node to the list
                node_idx = len(nodes)
                nodes.append(((weighted_y, weighted_x), node_idx))
                G.add_node(node_id, pos=(weighted_y, weighted_x))  # Add node to the graph
                node_id += 1

    G = connect_nodes_with_shared_block_neighbors(G, nodes, block_size)

    # show the original, binary images, and network
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title('Binary Image')
    # on the binary image, overplot a grid of block size x block size
    for i in range(0, img.shape[0], block_size):
        axs[1].axhline(i, color='red')
    for j in range(0, img.shape[1], block_size):
        axs[1].axvline(j, color='red')

    for node in nodes:
        axs[1].scatter(node[0][1], node[0][0], color='red')  # Note: (x, y) are reversed for scatter plot  
    # make sure same aspect ratio for this image and the network
    # Draw the network on the third subplot
    pos = {node: (x, -y) for (y, x), node in nodes}  # Adjust positions for matplotlib, invert y-axis
    nx.draw(G, pos, with_labels=True, node_color='red', node_size=50, edge_color='blue', ax=axs[2])

    # Set aspect of the third plot to be equal
    axs[2].set_aspect('equal')
    axs[2].set_title('Network')

    plt.savefig(f'results/binary_comp_{index}.png')
    plt.show()

    # Create a network based on the binary image
    return G, nodes

def connect_nodes_with_shared_block_neighbors(G, nodes, block_size):
    """Connect nodes in G that share neighbors in adjacent blocks."""
    def get_block_index(position, block_size):
        """Calculate the block index for a given position."""
        return (int(position[0] // block_size), int(position[1] // block_size))

    def find_neighbors_in_adjacent_blocks(node_index, nodes, block_size):
        """Find all nodes in adjacent blocks to a given node."""
        node_position = nodes[node_index][0]  # (y, x) position
        node_block_index = get_block_index(node_position, block_size)
        
        neighbors = []
        for i, ((y, x), _) in enumerate(nodes):
            other_block_index = get_block_index((y, x), block_size)
            if (abs(node_block_index[0] - other_block_index[0]) <= 1 and
                abs(node_block_index[1] - other_block_index[1]) <= 1 and
                i != node_index):  # Exclude the node itself
                neighbors.append(i)
        return neighbors
    
    for node_index in range(len(nodes)):
        # Find neighbors in adjacent blocks
        neighbor_indices = find_neighbors_in_adjacent_blocks(node_index, nodes, block_size)
        for neighbor_index in neighbor_indices:
            # Connect the current node with nodes in adjacent blocks if not already connected
            if not G.has_edge(node_index, neighbor_index):
                G.add_edge(node_index, neighbor_index)

    return G


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
    index = 14408
    img_data0 = np.array(X_train[index])  # Use .iloc[0] to access the first row for pandas DataFrame
    G, nodes = prep_img(img_data0, index)
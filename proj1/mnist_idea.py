# implement topological analysis on mnist dataset

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
import leidenalg as la
import igraph as ig
import time

def prep_img(img_data, index=None, digit_class = None, threshold=None, block_size=2, save_img=False):
    '''Converts image into network based on thresholding and block size'''
    img = img_data.reshape(28, 28)
    y, x =  np.where(img > 0)
    threshold_y = np.mean(y)
    threshold_x = np.mean(x)
    mean_bright = np.mean([threshold_y, threshold_x])
    # Thresholding
    if threshold is None:
        threshold = 240 - np.abs(mean_bright / 14)*52
    binary_image = img > threshold

    # Initialize the graph
    G = ig.Graph()

    nodes = []
    node_attributes = []  # Prepare to store node attributes

    for i in range(binary_image.shape[0] - block_size, -1, -block_size):
        if ((binary_image.shape[0] - i) // block_size) % 2 == 0:
            j_values = range(0, binary_image.shape[1], block_size)
        else:
            j_values = range(binary_image.shape[1] - block_size, -1, -block_size)
        
        for j in j_values:
            block = binary_image[i:i+block_size, j:j+block_size]
            if np.any(block):
                ys, xs = np.nonzero(block)
                ys = ys + i
                xs = xs + j
                weighted_y = np.mean(ys)
                weighted_x = np.mean(xs)

                # Store node attributes for later addition
                node_attributes.append({'pos': (weighted_y, weighted_x)})
                nodes.append((weighted_y, weighted_x))
                
                # Add node to the graph
                G.add_vertices(1)

    G = connect_nodes_with_shared_block_neighbors(G, nodes, block_size)
    G = G.spanning_tree(return_tree=True)

    # Assign attributes to nodes after all vertices and edges are added
    for idx, attrs in enumerate(node_attributes):
        G.vs[idx]['pos'] = attrs['pos']

    if save_img:

        # show the original, binary images, and network
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
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
            axs[1].scatter(node[1], node[0], color='red')  # Invert y-axis for correct orientation


        # simplify the network by removing leaf nodes
        # leaf_nodes = [node for node, degree in dict(G.degree()).items() if degree == 1]
        # G.remove_nodes_from(leaf_nodes)
    
        if index is not None and digit_class is not None:
            plt.title(f'Binary Image and Network for Image {index} (Class {digit_class})')
            plt.savefig(f'results/binary_comp_{index}_{digit_class}.pdf')
        else:
            # use time to save
            timestamp = time.time()
            plt.savefig(f'results/binary_comp_{timestamp}.pdf')
        plt.show()

    # Create a network based on the binary image
    return G, nodes

def connect_nodes_with_shared_block_neighbors(G, nodes, block_size):
    """Connect nodes in an igraph G that share neighbors in adjacent blocks."""
    
    def get_block_index(position, block_size):
        """Calculate the block index for a given position."""
        return (int(position[0] // block_size), int(position[1] // block_size))

    def find_neighbors_in_adjacent_blocks(node_index, nodes, block_size):
        """Find all nodes in adjacent blocks to a given node."""
        node_position = nodes[node_index]  # (y, x) position
        node_block_index = get_block_index(node_position, block_size)
        
        neighbors = []
        for i, (y, x) in enumerate(nodes):
            other_block_index = get_block_index((y, x), block_size)
            if (abs(node_block_index[0] - other_block_index[0]) <= 1 and
                abs(node_block_index[1] - other_block_index[1]) <= 1 and
                i != node_index):  # Exclude the node itself
                neighbors.append(i)
        return neighbors
    
    for node_index in range(len(nodes)):
        neighbor_indices = find_neighbors_in_adjacent_blocks(node_index, nodes, block_size)
        for neighbor_index in neighbor_indices:
            if not G.are_connected(node_index, neighbor_index):
                G.add_edges([(node_index, neighbor_index)])

    return G
def merge_close_nodes(G, nodes, distance_threshold=2):
    """
    Merge nodes that are closer than the distance_threshold.
    """
    # Find close nodes to be merged based on the distance threshold
    close_nodes = set()
    for i, ((y1, x1), idx1) in enumerate(nodes):
        for j, ((y2, x2), idx2) in enumerate(nodes):
            if i != j and np.linalg.norm([x1 - x2, y1 - y2]) < distance_threshold:
                close_nodes.add((min(idx1, idx2), max(idx1, idx2)))

    # Create a mapping from old to new nodes
    new_nodes = {}
    for idx1, idx2 in close_nodes:
        root1 = new_nodes.get(idx1, idx1)
        root2 = new_nodes.get(idx2, idx2)
        new_root = min(root1, root2)
        new_nodes[idx1] = new_root
        new_nodes[idx2] = new_root

    # Create a new graph with merged nodes
    new_G = nx.Graph()
    for old_idx, new_idx in new_nodes.items():
        new_G.add_node(new_idx, pos=nodes[old_idx][0])  # Use position of one of the old nodes

    # Add edges between new nodes in the new graph
    for (n1, n2) in G.edges():
        new_n1 = new_nodes.get(n1, n1)
        new_n2 = new_nodes.get(n2, n2)
        if new_n1 != new_n2:
            new_G.add_edge(new_n1, new_n2)

    # If there were nodes not close to any others, they won't be in the new_nodes map and need to be added to the new graph
    for ((y, x), idx) in nodes:
        if idx not in new_nodes:
            new_G.add_node(idx, pos=(y, x))

    return new_G

## run leiden algorithm ##
def run_leiden(G, index= None, digit_class = None, show_plot=False):
    '''run leiden algorithm on the graph'''
    partition = la.find_partition(G, la.ModularityVertexPartition)

    modularity = partition.modularity

    # get the clusters
    clusters = partition.membership

    # get the cluster sizes
    cluster_sizes = np.bincount(clusters)


    if show_plot:
        # color the nodes based on the cluster
        # Use igraph's plot function, specifying the BytesIO buffer as the target
        # and applying the defined visual style
        visual_style = {}
        visual_style["vertex_size"] = 20
        visual_style["bbox"] = (1000, 1000)
        visual_style["margin"] = 50
        visual_style["edge_width"] = 0.1  # Set edge width to a smaller value for thinner edges

        if index is not None and digit_class is not None:
            filename = f'results/leiden_{index}_{digit_class}.pdf'
            filename2 = f'results/leiden_{index}_{digit_class}_G.pdf'
        else:
            timestamp = time.time()
            filename = f'results/leiden_{timestamp}.pdf'

        ig.plot(partition, target=filename, **visual_style)
        ig.plot(G, target=filename2, **visual_style)

    return partition, clusters

## Load MNIST data ##
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

    index = 47
    digit=9
    G, nodes = prep_img(nines[index], index, digit_class=digit, save_img=True)
    partition, clusters = run_leiden(G, index, digit, show_plot=True)
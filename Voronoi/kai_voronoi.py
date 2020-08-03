import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial import Voronoi
import math
# https://stackoverflow.com/questions/48253800/shading-a-map-according-to-data-for-a-set-of-coordinates

def generate_voronoi_diagram(width, height, sparse_locations, sparse_data):
    '''
    sparse_locations should be list of centers with associated measurement locations (so for example one row could be [10,24])
    sparse_data should be measurements at sensor locations
    '''
    arr = np.zeros((width, height))
    imgx,imgy = width, height
    num_cells=np.shape(sparse_locations)[0]

    nx = sparse_locations[:,0]
    ny = sparse_locations[:,1]
    nr = list(range(num_cells))
    ng = nr
    nb = nr

    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i
            arr[x, y] = sparse_data[j]

    plt.imshow(arr)
    plt.show()

    return arr


def generate_weighted_voronoi_diagram(width, height, sparse_locations, sparse_data):
    '''
    sparse_locations should be list of centers with associated measurement locations (so for example one row could be [10,24])
    sparse_data should be measurements at sensor locations
    '''
    arr = np.zeros((width, height))
    imgx,imgy = width, height
    num_cells=np.shape(sparse_locations)[0]

    nx = sparse_locations[:,0]
    ny = sparse_locations[:,1]
    nr = list(range(num_cells))
    ng = nr
    nb = nr

    weighted_sparse_data = sparse_data/np.max(sparse_data)

    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < weighted_sparse_data[i]*dmin:
                    dmin = d
                    j = i
            arr[x, y] = sparse_data[j]

    plt.imshow(arr)
    plt.show()

    return arr

if __name__ == '__main__':
    # get random points
    np.random.seed(1234)

    # Using other (slower method)
    sparse_locations = np.random.randint(256,size=(15,2)) # 15 sensors
    sparse_data = np.random.uniform(low=0.0, high=1.0, size=(15,))
    
    input_array = generate_voronoi_diagram(256,256,sparse_locations,sparse_data)
    input_array = generate_weighted_voronoi_diagram(256,256,sparse_locations,sparse_data)    

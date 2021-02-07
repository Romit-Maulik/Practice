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

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def vorarr(regions, vertices, values, width, height, dpi=100):
    fig = plt.Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0,0,1,1])

    # colorize
    for region, value in zip(regions, values):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), color='blue', alpha=value)

    ax.plot(points[:,0], points[:,1], 'ko')
    ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

    canvas.draw()
    return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

# get random points
np.random.seed(1234)
points = np.random.rand(15, 2)
values = np.random.uniform(low=0.0, high=1.0, size=(len(points),))

# compute Voronoi tesselation
vor = Voronoi(points)

# voronoi_finite_polygons_2d function from https://stackoverflow.com/a/20678647/425458
regions, vertices = voronoi_finite_polygons_2d(vor)

# convert plotting data to numpy array
arr = vorarr(regions, vertices, values, width=128, height=128)
arr = arr[1:,1:]

# plot the numpy array
plt.imshow(np.sum(arr,axis=-1))
plt.show()


# Using other (slower method)
sparse_locations = np.random.randint(128,size=(15,2)) # 15 sensors
sparse_data = np.random.uniform(low=0.0, high=1.0, size=(15,))
generate_voronoi_diagram(128,128,sparse_locations,sparse_data)
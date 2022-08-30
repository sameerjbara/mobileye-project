import contextlib
import numpy as np
from matplotlib import pyplot as plt


def as_col(x):
    return np.ravel(x).reshape((-1, 1))


def as_row(x):
    return np.ravel(x).reshape((1, -1))


def padnan(x, y):
    return np.r_[as_row(x), as_row(y), as_row(x) + np.nan].T.ravel()


def points_inside(pts, polygon):
    """
    Returns a boolean array, array[i] == True means pts[i] is inside the polygon.
    Implemented with angle accumulation.
    :param pts: 2d points
    :param polygon: 2d polygon
    :return:
    """
    polygon = np.vstack((polygon, polygon[0, :]))  # close the polygon (if already closed shouldn't hurt)
    # assert np.all(polygon[0,:] == polygon[-1,:])  # polygon is expected to have first==last (document)
    sum_angles = np.zeros([len(pts), ])
    for i in range(len(polygon) - 1):
        v1 = polygon[i, :] - pts
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v1[norm_v1 == 0.0] = 1.0  # prevent divide-by-zero nans (entries will remain zero anyway)
        v1 = v1 / as_col(norm_v1)
        v2 = polygon[i + 1, :] - pts
        norm_v2 = np.linalg.norm(v2, axis=1)
        norm_v2[norm_v2 == 0.0] = 1.0  # prevent divide-by-zero nans (entries will remain zero anyway)
        v2 = v2 / as_col(norm_v2)
        dot_prods = np.sum(v1 * v2, axis=1)
        cross_prods = np.cross(v1, v2)
        angs = np.arccos(np.clip(dot_prods, -1, 1))
        angs = np.sign(cross_prods) * angs
        sum_angles += angs

    sum_angles = sum_angles * 180.0 / np.pi

    return abs(sum_angles) > 90.0


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()
    return plt.gca()


def get_stats(x, y, objects):
    """Get a list of points, and a list of polygons. Return which points are a hit, and which polygons are detected
    :param x: list of x values
    :param y: list of y values
    :param objects: list of dicts with 'polygon' to give a 2D list of vertices
    :return tuple: which_points_true, which_objects_detected"""
    pnts = np.c_[x, y]
    is_inside = np.zeros(len(pnts), dtype=bool)
    is_detected = []
    for obj in objects:
        pts_in_obj = points_inside(pnts, np.array(obj['polygon']))
        is_inside[pts_in_obj] = True
        is_detected.append(pts_in_obj.any())
    return is_inside, np.array(is_detected)


def get_interesting_gt_data(gt_data, what=None):
    """Retrieve the types we really want to measure"""
    if what is None:
        what = ['traffic light']
    objs = [o for o in gt_data['objects'] if o['label'] in what]
    return objs
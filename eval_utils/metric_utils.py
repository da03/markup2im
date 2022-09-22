import os
import cv2
import numpy as np
from PIL import Image
from tslearn.metrics import dtw, dtw_path, dtw_path_from_metric
from IPython import embed


def load_image_cv2(img_path: str) -> np.array:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

def load_image_pil(img_path: str) -> np.array:
    img = Image.open(img_path)
    return img

def get_column_features(img: np.array, th = 150):
    # check the image is grayscale
    assert(len(img.shape) == 2)
    features = []
    binary_img = (img < th).astype(int)
    #cv2.imwrite("img.png", img )
    cv2.imwrite("gt.png", binary_img * 255)
    for c in range(binary_img.shape[1]):
        features.append(binary_img[:, c])
    return features

def euclidean_dist_translation_invariant(x, y):
    '''
    allow 10% of vertical translation
    '''
    num_rows = x.shape[0]
    slack = int(np.floor(num_rows * 0.1))
    dists = []

    # no translation
    diff = (x - y) ** 2
    dist = np.sum(diff)
    dists.append(dist)
    
    if dist != 0.:
        # translate x downward (up to 10%)
        for s in range(1, slack+1):
            diff = (x[s:] - y[:-s]) ** 2
            dist = np.sum(diff)
            dists.append(dist)

        # translate y downward (up to 10%)
        for s in range(1, slack+1):
            diff = (x[:-s] - y[s:]) ** 2
            dist = np.sum(diff)
            dists.append(dist)

    return np.min(dists)

from numba import njit, prange
@njit()
def euclidean_dist_translation_invariant_faster(x, y):
    num_rows = x.shape[0]
    slack = int(np.floor(num_rows * 0.1))
    dists = []

    # no translation
    dist = 0.
    for di in range(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    dists.append(dist)

    if dist != 0.:
        # translate x downward (up to 10%)
        for s in range(1, slack+1):
            dist = 0.
            for di in range(x.shape[0]):
                if di+s < num_rows:
                    diff = (x[di+s] - y[di])
                    dist += diff * diff
            dists.append(dist)

        # translate y downward (up to 10%)
        for s in range(1, slack+1):
            dist = 0.
            for di in range(x.shape[0]):
                if di+s < num_rows:
                    diff = (x[di] - y[di+s])
                    dist += diff * diff
            dists.append(dist)

    return min(dists)

def dot_product(x, y):
    '''
    the difference from euclidiean distance is that each feature is normalized.
    '''
    cost = 0.
    return cost

def dot_product_translation_invariant(x, y):
    cost = 0.
    return cost

dist_metrics = {
    "euc_tsinv": euclidean_dist_translation_invariant_faster,
    "dot": dot_product,
    "dot_tsinv": dot_product_translation_invariant,
}

def calc_dtm_score(pred_img, gold_img, dist_metric=None):
    # get column fetures
    pred_feats = get_column_features(pred_img)
    gold_feats = get_column_features(gold_img)

    # calculate cost
    if dist_metric is None:
        path, cost = dtw_path(pred_feats, gold_feats)
    else:
        path, cost = dtw_path_from_metric(pred_feats, gold_feats, dist_metrics[dist_metric])
        if "euc" in dist_metric:
            cost = np.sqrt(cost)

    # normalize score between 0 and 1

    return cost
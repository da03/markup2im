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
    #cv2.imwrite("binary.png", binary_img * 255)
    for c in range(binary_img.shape[1]):
        features.append(binary_img[:, c])
    return features

def calc_dtm_score(pred_img, gold_img, dist_metric=None):
    # get column fetures
    pred_feats = get_column_features(pred_img)
    gold_feats = get_column_features(gold_img)
    
    # calculate features
    if dist_metric is None:
        path, cost = dtw_path(pred_feats, gold_feats)
    else:
        path, cost = dtw_path_from_metric(pred_feats, gold_feats, dist_metric)

    # normalize score between 0 and 1

    return cost
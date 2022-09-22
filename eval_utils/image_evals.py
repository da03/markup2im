# Test usage: python image_evals.py
# code reference: https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
from sewar.full_ref import rmse, ssim
from sewar.full_ref import psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

from typing import Dict
from IPython import embed

from clip_utils import clip_score
from metric_utils import load_image_cv2, load_image_pil, calc_dtm_score


def eval_image(pred_img_path: str, gold_img_path: str) -> Dict:
    evals = {}
    # load images
    pred_img_cv2, gold_img_cv2 = load_image_cv2(pred_img_path), load_image_cv2(gold_img_path)
    pred_img_pil, gold_img_pil = load_image_pil(pred_img_path), load_image_pil(gold_img_path)

    # calculate negated RMSE (↑)
    rmse_val = - rmse(pred_img_cv2, gold_img_cv2)
    evals["rmse"] = rmse_val

    # calculate Structural Similarity (↑)    
    ssim_val = ssim(pred_img_cv2, gold_img_cv2)
    evals["ssim"] = ssim_val

    # other metrics
    """
    evals["psnr"] = psnr(pred_img_cv2, gold_img_cv2)
    evals["uqi"] = uqi(pred_img_cv2, gold_img_cv2)
    evals["ergas"] = - ergas(pred_img_cv2, gold_img_cv2)
    evals["scc"] = scc(pred_img_cv2, gold_img_cv2)
    evals["rase"] = -rase(pred_img_cv2, gold_img_cv2)
    evals["sam"] = -sam(pred_img_cv2, gold_img_cv2)
    evals["msssim"] = msssim(pred_img_cv2, gold_img_cv2)
    evals["vifp"] = vifp(pred_img_cv2, gold_img_cv2)
    """

    # calculate CLIP distance (↑)   
    evals["clip"] = clip_score(pred_img_pil, gold_img_pil)

    # calculate custom metrics
    evals["dtw_euclidean"] = -calc_dtm_score(pred_img_cv2, gold_img_cv2)
    evals["dtw_euclidean_translation_invariant"] = -calc_dtm_score(pred_img_cv2, gold_img_cv2, "euc_tsinv")

    return evals

"""
def eval_image(pred_img_path: str, gold_img_path: str) -> Dict:
    evals = {}
    # load images
    pred_img_cv2, gold_img_cv2 = load_image_cv2(pred_img_path), load_image_cv2(gold_img_path)
    pred_img_pil, gold_img_pil = load_image_pil(pred_img_path), load_image_pil(gold_img_path)
    img_id = os.path.basename(gold_img_path).replace(".png", "")

    # calculate custom metrics
    evals["dtw_euclidean"] = -calc_dtm_score(pred_img_cv2, gold_img_cv2)
    evals["dtw_euclidean_translation_invariant"] = -calc_dtm_score(pred_img_cv2, gold_img_cv2, "euc_tsinv")

    return evals
"""


def visualize_ranked_results(sorted_dict: str, met_name: str, output_folder: str):
    # create visualization folder
    os.makedirs(output_folder, exist_ok=True)
    n = len(sorted_eval_dicts)

    fig, ax = plt.subplots(n, 2, figsize=(8, 1*n))
    for i, k in enumerate(sorted_dict.keys()):
        meta_data = k
        id, pred_img_path, gold_img_path = meta_data
        score = np.round(sorted_dict[k], 3)
        image = Image.open(gold_img_path)
        ax[i][0].imshow(image)
        ax[i][0].axis('off')
        image = Image.open(pred_img_path)
        ax[i][1].imshow(image)
        ax[i][1].axis('off')
        ax[i][1].set_title("{} ".format(score))
        #ax[i][1].set_title("{} gt: {}".format(id, score))

    fig_name = os.path.join(output_folder, f"{met_name}.png")
    plt.savefig(fig_name)
    plt.close()

if __name__ == "__main__":
    # testing out image eval functions
    pred_img_folder =  "../images_rendered/images_pred" # "../images_rendered_html/images_pred"
    pred_img_pathes = [os.path.join(pred_img_folder, f) for f in os.listdir(pred_img_folder) if ".png" in f]
    gold_img_folder =  "../images_rendered/images_gold" #"../images_rendered_html/images_gold"
    output_folder = "html_visualization"
    
    results = []
    for pred_img_path in pred_img_pathes:
        img_id = os.path.basename(pred_img_path).replace(".png", "")
        gold_img_path = "{}/{}.png".format(gold_img_folder, img_id)
        eval = eval_image(pred_img_path, gold_img_path)
        eval["meta_data"] = (img_id, pred_img_path, gold_img_path)
        results.append(eval)
        print("{}: {}".format(img_id, eval))

    # create ranking according to metrics
    for met_name in ["rmse", "ssim", "clip", "psnr", "uqi", "ssim", "ergas", "scc", "rase", "sam", "vifp", "dtw_euclidean", "dtw_euclidean_translation_invariant"]:
        eval_dicts = {}
        for r in results:
            if met_name in r:
                eval_dicts[r["meta_data"]] = r[met_name]
        
        if len(eval_dicts) > 0:
            sorted_eval_dicts = {k: v for k, v in sorted(eval_dicts.items(), key=lambda item: item[1], reverse=True)}
            visualize_ranked_results(sorted_eval_dicts, met_name, output_folder)

    


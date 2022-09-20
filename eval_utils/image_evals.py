# Test usage: python image_evals.py
# code reference: https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from sewar.full_ref import rmse, ssim
from sewar.full_ref import psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

from typing import Dict
from IPython import embed

from load_clip import calc_clip_logits

def load_image_cv2(img_path: str) -> np.array:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

def load_image_pil(img_path: str) -> np.array:
    img = Image.open(img_path)
    return img

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
    evals["psnr"] = psnr(pred_img_cv2, gold_img_cv2)
    evals["uqi"] = uqi(pred_img_cv2, gold_img_cv2)
    evals["ergas"] = - ergas(pred_img_cv2, gold_img_cv2)
    evals["scc"] = scc(pred_img_cv2, gold_img_cv2)
    evals["rase"] = -rase(pred_img_cv2, gold_img_cv2)
    evals["sam"] = -sam(pred_img_cv2, gold_img_cv2)
    #evals["msssim"] = msssim(pred_img_cv2, gold_img_cv2)
    evals["vifp"] = vifp(pred_img_cv2, gold_img_cv2)

    # calculate CLIP distance (↑)   
    evals["clip"] = calc_clip_logits(pred_img_pil, gold_img_pil)

    return evals
"""
def visualize_ranked_results(sorted_dict: str, met_name: str):
    # create visualization folder
    embed()
    folder_name = "visualization"
    os.makedirs(folder_name, exist_ok=True)

    n = len(sorted_eval_dicts)
    plt.figure(figsize=(8, 8*n))

    for i, k in enumerate(sorted_dict.keys()):
        meta_data = k
        id, pred_img_path, gold_img_path = meta_data
        score = np.round(sorted_dict[k], 3)

        # Debug, plot figure
        ax1 = plt.subplot(n, 2, i + 1)
        ax1.axis('off')
        ax1.set_title("{} pred: {}".format(id, score))
        ax1.imshow(mpimg.imread(pred_img_path))
        ax2 = plt.subplot(n, 2, i + 2)
        ax2.axis('off')
        ax2.set_title("{} gt: {}".format(id, score))
        ax2.imshow(mpimg.imread(gold_img_path))
    
    fig_name = os.path.join(folder_name, f"{met_name}.png")
    plt.savefig(fig_name)
    plt.close()
"""

def visualize_ranked_results(sorted_dict: str, met_name: str):
    # create visualization folder
    folder_name = "visualization"
    os.makedirs(folder_name, exist_ok=True)
    n = len(sorted_eval_dicts)

    fig, ax = plt.subplots(n, 2, figsize=(8, 1*n))
    for i, k in enumerate(sorted_dict.keys()):
        meta_data = k
        id, pred_img_path, gold_img_path = meta_data
        score = np.round(sorted_dict[k], 3)
        image = Image.open(gold_img_path)
        ax[i][0].imshow(image)
        ax[i][0].axis('off')
        #ax[i][0].set_title("{} pred: {}".format(id, score))
        image = Image.open(pred_img_path)
        ax[i][1].imshow(image)
        ax[i][1].axis('off')
        ax[i][1].set_title("{} ".format(score))
        #ax[i][1].set_title("{} gt: {}".format(id, score))

    fig_name = os.path.join(folder_name, f"{met_name}.png")
    plt.savefig(fig_name)
    plt.close()


if __name__ == "__main__":
    # testing out image eval functions
    pred_img_folder = "../images_rendered/images_pred"
    pred_img_pathes = [os.path.join(pred_img_folder, f) for f in os.listdir(pred_img_folder) if ".png" in f]
    gold_img_folder = "../images_rendered/images_gold"
    gold_img_pathes = [os.path.join(gold_img_folder, f)for f in os.listdir(gold_img_folder) if ".png" in f]
    
    results = []

    for pred_img_path, gold_img_path in zip(pred_img_pathes, gold_img_pathes):
        img_id = os.path.basename(pred_img_path).replace(".png", "")
        eval = eval_image(pred_img_path, gold_img_path)
        eval["meta_data"] = (img_id, pred_img_path, gold_img_path)
        results.append(eval)
        print("{}: {}".format(img_id, eval))

    # create ranking according to metrics
    for met_name in ["rmse", "ssim", "clip", "psnr", "uqi", "ssim", "ergas", "scc", "rase", "sam", "vifp"]:
        eval_dicts = {}
        for r in results:
            eval_dicts[r["meta_data"]] = r[met_name]
        
        sorted_eval_dicts = {k: v for k, v in sorted(eval_dicts.items(), key=lambda item: item[1], reverse=True)}
        visualize_ranked_results(sorted_eval_dicts, met_name)

    


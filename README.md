## Dependencies
`pip install -r HF_Space_App/requirements.txt`

## Sample Images
Sample images can be found at `images_rendered`, which contains two folders: `images_gold` contains the ground truth images, and `images_pred` contains the predicted images. The goal of evaluation is to compare the images with the same names in both folders.

## Generate Images 
To generate the folder `images_rendered`, run the following command (on a machine with GPUs)
```
CUDA_VISIBLE_DEVICES=0 python test_inf.py
```

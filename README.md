# Markup-to-Image Diffusion Models with Scheduled Sampling

We provide code to reproduce [our paper on markup-to-image generation](https://arxiv.org/pdf/2210.05147.pdf). Our code is built on top of HuggingFace [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers).

## Online Demo: [https://huggingface.co/spaces/yuntian-deng/latex2im](https://huggingface.co/spaces/yuntian-deng/latex2im).

## Example Generations

Scheduled Sampling            |   Baseline                              |        Ground Truth      |
:----------------------------:|:---------------------------------------:|:------------------------:|
![](imgs/math_rendering.gif)  |  ![](imgs/math_rendering_baseline.gif)  | ![](imgs/433d71b530.png) |
![](imgs/tables_rendering.gif)|  ![](imgs/tables_rendering_baseline.gif)| ![](imgs/42725-full.png) |
![](imgs/music_rendering.gif) |  ![](imgs/music_rendering_baseline.gif) | ![](imgs/comp.17342.png) |
![](imgs/molecules_rendering.gif)|  ![](imgs/molecules_rendering_baseline.gif)| ![](imgs/B-1173.png) |

## Prerequisites

* [Pytorch](https://pytorch.org/get-started/locally/)

```
pip install transformers
pip install accelerate
pip install -qU git+https://github.com/da03/diffusers
```

## Datasets & Pretrained Models

All datasets have been uploaded to [Huggingface datasets](https://huggingface.co/yuntian-deng).

* Math: [data](https://huggingface.co/datasets/yuntian-deng/im2latex-100k) [model](models/math/scheduled_sampling/model_e100_lr0.0001.pt.100)
* Simple Tables: [data](https://huggingface.co/datasets/yuntian-deng/im2html-100k) [model](models/tables/scheduled_sampling/model_e100_lr0.0001.pt.100)
* Sheet Music: [data](https://huggingface.co/datasets/yuntian-deng/im2ly-35k-syn) [model](music/math/scheduled_sampling/model_e100_lr0.0001.pt.100)
* Molecules: [data](https://huggingface.co/datasets/yuntian-deng/im2smiles-20k) [model](models/molecules/scheduled_sampling/model_e100_lr0.0001.pt.100)

## Usage

### Training

#### Math

To train the diffusion model,

```
python src/train.py --save_dir models/math
```
#### Tables

To train the diffusion model,

```
python src/train.py --dataset_name yuntian-deng/im2html-100k --save_dir models/tables 
```

#### Music

In our paper, we trained on the music dataset with 4 A100 GPUs. You might need to tune `--batch_size` and  `--gradient_accumulation_steps` if you want to use a single GPU to train or if your GPUs have less memory.

We first run

```
accelerate config
```
to use 4 GPUs on a single machine. Note that we did not use fp16 or DeepSpeed.

Next, we launch multi-GPU training using accelerate:

```
accelerate launch src/train.py --dataset_name yuntian-deng/im2ly-35k-syn --save_dir models/music
```

#### Molecules

To train the diffusion model,

```
python src/train.py --dataset_name yuntian-deng/im2smiles-20k --save_dir models/molecules
```

### Generation

#### Math

To generate,

```
python scripts/generate.py --model_path models/math/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/math --save_intermediate_every -1
```

#### Tables

To generate,

```
python scripts/generate.py --dataset_name yuntian-deng/im2html-100k --model_path models/tables/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/tables --save_intermediate_every -1
```

#### Music

To generate,

```
python scripts/generate.py --dataset_name yuntian-deng/im2ly-35k-syn --model_path models/music/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/music --save_intermediate_every -1
```

#### Molecules

To generate,

```
python scripts/generate.py --dataset_name yuntian-deng/im2smiles-20k --model_path models/molecules/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/molecules --save_intermediate_every -1
```

### Visualization

#### Math

To visualize the generation process, we need to first use the following command to save the intermediate images during generation:

```
python scripts/generate.py --model_path models/math/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/math/scheduled_sampling_visualization --save_intermediate_every 1 --num_batches 1
```

Next, we put together a gif image from the generated images:

```
python scripts/make_gif.py --input_dir outputs/math/scheduled_sampling_visualization/ --output_filename imgs/math_rendering.gif --select_filename 433d71b530.png --show_every 10
```

We can similarly visualize results from the baseline.

```
python scripts/generate.py --model_path models/math/baseline/model_e100_lr0.0001.pt.100 --output_dir outputs/math/baseline_visualization --save_intermediate_every 1 --num_batches 1
```

```
python scripts/make_gif.py --input_dir outputs/math/baseline_visualization/ --output_filename imgs/math_rendering_baseline.gif --select_filename 433d71b530.png --show_every 10
```

#### Tables

To visualize the generation process, we need to first use the following command to save the intermediate images during generation:

```
python scripts/generate.py --dataset_name yuntian-deng/im2html-100k --model_path models/tables/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/tables/scheduled_sampling_visualization --save_intermediate_every 1 --num_batches 1
```

Next, we put together a gif image from the generated images:

```
python scripts/make_gif.py --input_dir outputs/tables/scheduled_sampling_visualization/ --output_filename imgs/tables_rendering.gif --select_filename 42725-full.png --show_every 10
```

We can similarly visualize results from the baseline.

```
python scripts/generate.py --dataset_name yuntian-deng/im2html-100k --model_path models/tables/baseline/model_e100_lr0.0001.pt.100 --output_dir outputs/tables/baseline_visualization --save_intermediate_every 1 --num_batches 1
```

```
python scripts/make_gif.py --input_dir outputs/tables/baseline_visualization/ --output_filename imgs/tables_rendering_baseline.gif --select_filename 42725-full.png --show_every 10
```

#### Music

To visualize the generation process, we need to first use the following command to save the intermediate images during generation:

```
python scripts/generate.py --dataset_name yuntian-deng/im2ly-35k-syn --model_path models/music/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/music/scheduled_sampling_visualization --save_intermediate_every 1 --num_batches 1
```

Next, we put together a gif image from the generated images:

```
python scripts/make_gif.py --input_dir outputs/music/scheduled_sampling_visualization/ --output_filename imgs/music_rendering.gif --select_filename comp.17342.png --show_every 10
```

We can similarly visualize results from the baseline.

```
python scripts/generate.py --dataset_name yuntian-deng/im2ly-35k-syn --model_path models/music/baseline/model_e100_lr0.0001.pt.100 --output_dir outputs/music/baseline_visualization --save_intermediate_every 1 --num_batches 1
```

```
python scripts/make_gif.py --input_dir outputs/music/baseline_visualization/ --output_filename imgs/music_rendering_baseline.gif --select_filename comp.17342.png --show_every 10
```

#### Molecules

To visualize the generation process, we need to first use the following command to save the intermediate images during generation:

```
python scripts/generate.py --dataset_name yuntian-deng/im2smiles-20k --model_path models/molecules/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/molecules/scheduled_sampling_visualization --save_intermediate_every 1 --num_batches 1
```

Next, we put together a gif image from the generated images:

```
python scripts/make_gif.py --input_dir outputs/molecules/scheduled_sampling_visualization/ --output_filename imgs/molecules_rendering.gif --select_filename B-1173.png --show_every 10
```

We can similarly visualize results from the baseline.

```
python scripts/generate.py --dataset_name yuntian-deng/im2smiles-20k --model_path models/molecules/baseline/model_e100_lr0.0001.pt.100 --output_dir outputs/molecules/baseline_visualization --save_intermediate_every 1 --num_batches 1
```

```
python scripts/make_gif.py --input_dir outputs/molecules/baseline_visualization/ --output_filename imgs/molecules_rendering_baseline.gif --select_filename B-1173.png --show_every 10
```

## Citation

```
@misc{deng2022markuptoimage,
      title={Markup-to-Image Diffusion Models with Scheduled Sampling}, 
      author={Yuntian Deng and Noriyuki Kojima and Alexander M. Rush},
      year={2022},
      eprint={2210.05147},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

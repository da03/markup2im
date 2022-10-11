# Markup-to-Image Diffusion Models with Scheduled Sampling

We provide code to reproduce our results on markup-to-image generation. Our code is built on top of HuggingFace [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers).

## Online Demo

An online demo can be found at [https://huggingface.co/spaces/yuntian-deng/latex2im](https://huggingface.co/spaces/yuntian-deng/latex2im).

## Generation Examples

Scheduled Sampling            |   Baseline
:----------------------------:|:---------------------------------------:
![](imgs/math_rendering.gif)  |  ![](imgs/math_rendering_baseline.gif)

![](table_rendering.gif)

![](music_rendering.gif)

![](molecule_rendering.gif)



## Prerequisites

* [Pytorch](https://pytorch.org/get-started/locally/)

```
pip install transformers
pip install -qU git+https://github.com/da03/diffusers
```

## Datasets & Pretrained Models

All datasets have been uploaded to [Huggingface datasets](https://huggingface.co/yuntian-deng).

* Math: [data]() [model]()
* Simple Tables: [data]() [model]()
* Sheet Music: [data]() [model]()
* Molecules: [data]() [model]()

## Usage

### Training


### Generation

### Visualizations

```
python scripts/visualize_intermediate_steps.py --model_path models/latex/scheduled_sampling/model_e100_lr0.0001.pt.100 --output_dir outputs/latex/scheduled_sampling_visualization --save_intermediate_every 1
```

```
python scripts/visualize_intermediate_steps.py --model_path models/latex/baseline/model_e100_lr0.0001.pt.100 --output_dir outputs/latex/baseline_visualization --save_intermediate_every 1
```

## Citation


# Markup-to-Image Diffusion Models with Scheduled Sampling

![](imgs/math_rendering.gif)

![](table_rendering.gif)

![](music_rendering.gif)

![](molecule_rendering.gif)

Here we provide code to reproduce our results. Our code is built on top of [HuggingFace diffusers](https://github.com/huggingface/diffusers) and [HuggingFace transformers](https://github.com/huggingface/transformers).

An online demo of latex rendering can be found at [https://huggingface.co/spaces/yuntian-deng/latex2im](https://huggingface.co/spaces/yuntian-deng/latex2im).

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


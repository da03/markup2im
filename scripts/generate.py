import math
import random
import sys
import os
import torch
import tqdm
import argparse
import torch.nn
import numpy as np

from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from accelerate import Accelerator

sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../src/'))
from markup2im_constants import get_image_size, get_input_field, get_encoder_model_type, get_color_mode
from markup2im_models import create_image_decoder, encode_text

#torch.backends.cuda.matmul.allow_tf32=True

def process_args(args):
    parser = argparse.ArgumentParser(description='Generate from a trained model.')

    parser.add_argument('--dataset_name',
                        type=str, default='yuntian-deng/im2latex-100k',
                        help=('Specifies which dataset to use.'
                        ))
    parser.add_argument('--model_path',
                        type=str, default='models/latex/scheduled_sampling/model_e100_lr0.0001.pt.100',
                        help=('Specifies which trained model to decode from.'
                        ))
    parser.add_argument('--input_field',
                        type=str, default=None,
                        help=('Field in the dataset containing input markups. If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--color_mode',
                        type=str, default=None,
                        help=('Specifies grayscale (grayscale) or RGB (rgb). If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--encoder_model_type',
                        type=str, default=None,
                        help=('Specifies encoder model type. If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--image_height',
                        type=int, default=None,
                        help=('Specifies the height of images to generate. If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--image_width',
                        type=int, default=None,
                        help=('Specifies the width of images to generate. If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--select_filename',
                        type=str, default='None',
                        help=('Only run inference on examples with this filename.'
                        ))
    parser.add_argument('--output_dir',
                        type=str, required=True,
                        help=('Output directory.'
                        ))
    parser.add_argument('--split',
                        type=str, default='test',
                        help=('Dataset split.'
                        ))
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help=('Batch size.'
                        ))
    parser.add_argument('--num_batches',
                        type=int, default=-1,
                        help=('Number of batches to decode. To generate all batches, set it to -1.'
                        ))
    parser.add_argument('--max_input_length',
                        type=int, default=1024,
                        help=('Max input length. Longer inputs will be truncated.'
                        ))
    parser.add_argument('--save_intermediate_every',
                        type=int, default=-1,
                        help=('Saves intermediate diffusion steps every this many steps. When <0 does not save any intermediate images.'
                        ))
    parser.add_argument('--seed1',
                        type=int, default=42,
                        help=('Random seed for shuffling data. Shouldn\'t be changed to be comparable to numbers reported in the paper.'
                        ))
    parser.add_argument('--seed2',
                        type=int, default=1234,
                        help=('Random seed for data loader. Shouldn\'t be changed to be comparable to numbers reported in the paper.'
                        ))
    parameters = parser.parse_args(args)
    return parameters

def load_pipeline(image_decoder, model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    state_dict_new = {}
    for k in state_dict:
        k_out = k.replace('module.', '')
        state_dict_new[k_out] = state_dict[k]
    image_decoder.load_state_dict(state_dict_new)
    
    accelerator = Accelerator(mixed_precision='no')
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
    pipeline = DDPMPipeline(unet=image_decoder, scheduler=noise_scheduler)
    return pipeline


def evaluate(dataloader, tokenizer, text_encoder, pipeline, output_dir, num_batches, save_intermediate_every=-1):
    gold_dir = os.path.join(output_dir, "images_gold")
    os.makedirs(gold_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "images_pred")
    os.makedirs(pred_dir, exist_ok=True)
    for step, batch in tqdm.tqdm(enumerate(dataloader)):
        gold_images = batch['gold_images']
        filenames = batch['filenames']
        input_ids = batch['input_ids'].cuda()
        masks = batch['attention_mask'].cuda()
        encoder_hidden_states = encode_text(text_encoder, input_ids, masks)
        for iii, input_id in enumerate(input_ids):
            formula = tokenizer.decode(input_id, skip_special_symbols=True).replace('<|endoftext|>', '')
            print (f'{iii:04d}: {formula}')
            print ()
        swap_step = -1
        t = 0
        for _, pred_images in pipeline.run_clean(
            batch_size = input_ids.shape[0],
            generator=torch.manual_seed(0),
            encoder_hidden_states = encoder_hidden_states,
            attention_mask=masks,
            swap_step=swap_step,
            ):
            pred_images = pipeline.numpy_to_pil(pred_images)
            if save_intermediate_every > 0:
                if t % save_intermediate_every == 0:
                    for filename, gold_image, pred_image in zip(filenames, gold_images, pred_images):
                        pred_image.save(os.path.join(pred_dir, filename + f'_{t:04d}.png'))
            t += 1

        for filename, gold_image, pred_image in zip(filenames, gold_images, pred_images):
            gold_image.save(os.path.join(gold_dir, filename))
            pred_image.save(os.path.join(pred_dir, filename))
        if step == num_batches-1:
            break

        print ('='*10)

def main(args):
    # Get default arguments
    if (args.image_height is not None) and (args.image_width is not None):
        image_size = (args.image_height, args.image_width)
    else:
        print (f'Using default image size for dataset {args.dataset_name}')
        image_size = get_image_size(args.dataset_name)
        print (f'Default image size: {image_size}')
    args.image_size = image_size
    if args.input_field is not None:
        input_field = args.input_field
    else:
        print (f'Using default input field for dataset {args.dataset_name}')
        input_field = get_input_field(args.dataset_name)
        print (f'Default input field: {input_field}')
    args.input_field = input_field
    if args.encoder_model_type is not None:
        encoder_model_type = args.encoder_model_type
    else:
        print (f'Using default encoder model type for dataset {args.dataset_name}')
        encoder_model_type = get_encoder_model_type(args.dataset_name)
        print (f'Default encoder model type: {encoder_model_type}')
    args.encoder_model_type = encoder_model_type
    if args.color_mode is not None:
        color_mode = args.color_mode
    else:
        print (f'Using default color mode for dataset {args.dataset_name}')
        color_mode = get_color_mode(args.dataset_name)
        print (f'Default color mode: {color_mode}')
    args.color_mode = color_mode 
    assert args.color_mode in ['grayscale', 'rgb']
    if args.color_mode == 'grayscale':
        args.color_channels = 1
    else:
        args.color_channels = 3

    # Load data
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.shuffle(seed=args.seed1)
   
    # Filter data (such as 433d71b530.png)
    if args.select_filename.lower() != 'none':
        print (f'Only running inference on {args.select_filename}')
        dataset = dataset.filter(lambda x: x['filename'] == args.select_filename)

    # Load input tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_type)

    # Load input encoder
    text_encoder = AutoModel.from_pretrained(args.encoder_model_type).cuda()
  
    # Preprocess data to form batches
    def preprocess_formula(formula):
      example = tokenizer(formula, truncation=True, max_length=args.max_input_length)
      input_ids = example['input_ids']
      attention_mask = example['attention_mask']
      return input_ids, attention_mask
    
    def transform(examples):
        gold_images = [image for image in examples["image"]]
        formulas_and_masks = [preprocess_formula(formula) for formula in examples[args.input_field]]
        formulas = [item[0] for item in formulas_and_masks]
        masks = [item[1] for item in formulas_and_masks]
        filenames = examples['filename']
        return {'input_ids': formulas, 'attention_mask': masks, 'filenames': filenames, 'gold_images': gold_images}
    
    dataset.set_transform(transform)

    def collate_fn(examples):
        eos_id = tokenizer.encode(tokenizer.eos_token)[0] # legacy code, might be unnecessary
        max_len = max([len(example['input_ids']) for example in examples]) + 1
        examples_out = []
        for example in examples:
            example_out = {}
            orig_len = len(example['input_ids'])
            formula = example['input_ids'] + [eos_id,] * (max_len - orig_len)
            example_out['input_ids'] = torch.LongTensor(formula)
            attention_mask = example['attention_mask'] + [1,] + [0,] * (max_len - orig_len - 1)
            example_out['attention_mask'] = torch.LongTensor(attention_mask)
            #example_out['images'] = example['images']
            examples_out.append(example_out)
        batch = default_collate(examples_out)
        filenames = [example['filenames'] for example in examples]
        gold_images = [example['gold_images'] for example in examples]
        batch['filenames'] = filenames
        batch['gold_images'] = gold_images 
        return batch
    
    torch.manual_seed(args.seed2)
    random.seed(args.seed2)
    np.random.seed(args.seed2)

    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, worker_init_fn=np.random.seed(0), num_workers=0)

    # Create and load models
    text_encoder = AutoModel.from_pretrained(args.encoder_model_type).cuda()
    # forward a fake batch to figure out cross_attention_dim
    hidden_states = encode_text(text_encoder, torch.zeros(1,1).long().cuda(), None)
    cross_attention_dim = hidden_states.shape[-1]
    
    image_decoder = create_image_decoder(image_size=args.image_size, color_channels=args.color_channels, \
            cross_attention_dim=cross_attention_dim)
    image_decoder = image_decoder.cuda()
    pipeline = load_pipeline(image_decoder, args.model_path)
    
    evaluate(eval_dataloader, tokenizer, text_encoder, pipeline, args.output_dir, args.num_batches, args.save_intermediate_every)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    main(args)

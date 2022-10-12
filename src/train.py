import os
import sys
import random
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm

sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../src/'))
from markup2im_constants import get_image_size, get_input_field, get_encoder_model_type, get_color_mode
from markup2im_models import create_image_decoder, encode_text, save_model


torch.backends.cuda.matmul.allow_tf32=True # for speed


def process_args(args):
    parser = argparse.ArgumentParser(description='Train a diffusion model.')

    parser.add_argument('--dataset_name',
                        type=str, default='yuntian-deng/im2latex-100k',
                        help=('Specifies which dataset to use.'
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
    parser.add_argument('--save_dir',
                        type=str, required=True,
                        help=('Output directory for saving model checkpoints.'
                        ))
    parser.add_argument('--split',
                        type=str, default='train',
                        help=('Dataset split.'
                        ))
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help=('Batch size.'
                        ))
    parser.add_argument('--gradient_accumulation_steps',
                        type=int, default=1,
                        help=('Gradient accumulation steps.'
                        ))
    parser.add_argument('--num_epochs',
                        type=int, default=100,
                        help=('Number of epochs.'
                        ))
    parser.add_argument('--scheduled_sampling_weights_start',
                        type=float, nargs='+', default=[0,],
                        help=("""The starting weight of applying scheduled sampling. 
                        Pass in a list for higher-order scheduled sampling (higher $m$ where $m$ denotes the number
                        of rollout steps, see our paper for more details). For example,
                        `--scheduled_sampling_weights_start 0 --scheduled_sampling_weights_start 0.5` will start from 
                        0 weight of applying first-order (m=1) scheduled sampling and 0.5 of applying second-order (m=2)
                        scheduled sampling."""
                        ))
    parser.add_argument('--scheduled_sampling_weights_end',
                        type=float, nargs='+', default=[0.5,],
                        help=("""The end weight of applying scheduled sampling. 
                        Pass in a list for higher-order (higher number of rollout steps) scheduled sampling. For example,
                        `--scheduled_sampling_weights_end 0.2 --scheduled_sampling_weights_end 0.3` will end with
                        0.2 weight of applying first-order (m=1) scheduled sampling and 0.3 of applying second-order (m=2)
                        scheduled sampling."""
                        ))
    parser.add_argument('--min_scheduled_sampling_step',
                        type=int, default=50,
                        help=('Do not apply scheduled sampling if the number of steps is smaller than this to avoid loss blowing up.'
                        ))
    parser.add_argument('--learning_rate',
                        type=int, default=1e-4,
                        help=('Learning rate.'
                        ))
    parser.add_argument('--lr_warmup_steps',
                        type=int, default=500,
                        help=('Lr warmup steps.'
                        ))
    parser.add_argument('--clip_grad_norm',
                        type=float, default=1.0,
                        help=('Clip gradient norm.'
                        ))
    parser.add_argument('--save_model_every',
                        type=int, default=5,
                        help=('Saves intermediate model checkpoints every this many steps.'
                        ))
    parser.add_argument('--mixed_precision',
                        type=str, default='no',
                        help=('Can be fp16 or no (fp32).'
                        ))
    parser.add_argument('--max_input_length',
                        type=int, default=1024,
                        help=('Max input length. Longer inputs will be truncated.'
                        ))
    parser.add_argument('--num_dataloader_workers',
                        type=int, default=1,
                        help=('Number of workers for dataloder.'
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

def train(train_dataloader, save_dir, save_model_every, \
        text_encoder, image_decoder, noise_scheduler, \
        scheduled_sampling_weights_start, scheduled_sampling_weights_end, \
        min_scheduled_sampling_step, \
        optimizer, lr_scheduler, num_epochs, gradient_accumulation_steps=1, \
        clip_grad_norm=1.0, learning_rate=1e-4, \
        mixed_precision='no'):
    #import pdb; pdb.set_trace()
    learning_rate = optimizer.defaults['lr']
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        logging_dir=os.path.join(save_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("markup2im_train")
    
    image_decoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        image_decoder, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    for epoch in range(num_epochs):
        m_probs = []
        # the weights of scheduled sampling change linearly throughout training
        for prob_start_m, prob_end_m in zip(scheduled_sampling_weights_start, scheduled_sampling_weights_end):
            prob_m = prob_start_m + epoch / num_epochs * (prob_end_m - prob_start_m)
            m_probs.append(prob_m)
        m_probs.insert(0, 1-sum(m_probs)) # the weight of not applying scheduled sampling
        acc_probs = [] # accumulated probabilities
        acc_prob = 0
        for p in m_probs:
            acc_prob += p
            acc_probs.append(acc_prob)
        print ('='*10)
        disp_str = ' '.join([f'{i} ({m_probs[i]})' for i in range(len(m_probs))])
        print (f'probs of applying rollout m: {disp_str}')
        print (f'acc probs: {acc_probs}')
        print ('='*10)
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(accelerator.device)
            input_ids = batch['input_ids'].to(accelerator.device)
            masks = batch['attention_mask'].to(accelerator.device)
            encoder_hidden_states = encode_text(text_encoder, input_ids, masks)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Sample m in scheduled sampling according to m_probs
            # Note that we use the same m per batch on the same device
            p = random.random()
            for m in range(len(m_probs)):
                acc_prob = acc_probs[m]
                if p < acc_prob:
                    break

            # Skip scheduled sampling if t is too small to avoid loss blowing up
            if min(timesteps) <= min_scheduled_sampling_step:
                m = 0
            # If there's not enough number of steps then decrease m
            while max(timesteps) >= noise_scheduler.num_train_timesteps-m:
                m -= 1

            # find input to the diffusion model
            with torch.no_grad():
                # first, sample t + m 
                noisy_images_t_plus_m = noise_scheduler.add_noise(clean_images, noise, timesteps+m)
                noisy_images_t_plus_s = noisy_images_t_plus_m
                # next, decode and clean
                for s in range(m):
                    # predict noise
                    noise_pred_rollback_s = image_decoder(noisy_images_t_plus_s, timesteps+m-s, encoder_hidden_states, attention_mask=masks)["sample"]
                    lambs_s, alpha_prod_ts_s = noise_scheduler.get_lambda_and_alpha(timesteps+m-s)
                    # clean img predicted
                    x_0_pred = (noisy_images_t_plus_s - lambs_s.view(-1, 1, 1, 1) * noise_pred_rollback_s) / alpha_prod_ts_s.view(-1, 1, 1, 1)
                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                    # get previous step sample
                    noisy_images_t_plus_s_minus_one  = noise_scheduler.add_noise(x_0_pred, noise, timesteps + m-s-1)
                    # update
                    noisy_images_t_plus_s = noisy_images_t_plus_s_minus_one
                noisy_images_t = noisy_images_t_plus_s

            with accelerator.accumulate(image_decoder):
                # Predict the noise residual
                noise_pred = image_decoder(noisy_images_t, timesteps, encoder_hidden_states, attention_mask=masks)["sample"]
                lambs_t, alpha_prod_ts_t = noise_scheduler.get_lambda_and_alpha(timesteps)
                noise = (noisy_images_t - alpha_prod_ts_t.view(-1, 1, 1, 1) * clean_images) / lambs_t.view(-1, 1, 1, 1)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(image_decoder.parameters(), clip_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()*gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        if epoch % save_model_every == 0:
            save_model(image_decoder, os.path.join(save_dir, f'model_e{num_epochs}_lr{learning_rate}.pt.{epoch}'))


def main(args):
    # Check arguments
    assert len(args.scheduled_sampling_weights_start) == len(args.scheduled_sampling_weights_end)
    assert all([0 <= item <= 1 for item in args.scheduled_sampling_weights_start])
    assert all([0 <= item <= 1 for item in args.scheduled_sampling_weights_end])
    assert sum(args.scheduled_sampling_weights_start) <= 1
    assert sum(args.scheduled_sampling_weights_end) <= 1
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
   
    # Load input tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_type)

    # Load input encoder
    text_encoder = AutoModel.from_pretrained(args.encoder_model_type).cuda()
  
    # Preprocess data to form batches
    transform_list = []
    if args.color_mode == 'grayscale':
        transform_list.append(transforms.Grayscale(num_output_channels=args.color_channels))
    preprocess_image = transforms.Compose(
        transform_list + [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def preprocess_formula(formula):
        example = tokenizer(formula, truncation=True, max_length=args.max_input_length)
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        return input_ids, attention_mask
    
    def transform(examples):
        images = [preprocess_image(image.convert("RGB")) for image in examples["image"]]
        gold_images = [image for image in examples["image"]]
        formulas_and_masks = [preprocess_formula(formula) for formula in examples[args.input_field]]
        formulas = [item[0] for item in formulas_and_masks]
        masks = [item[1] for item in formulas_and_masks]
        filenames = examples['filename']
        return {'images': images, 'input_ids': formulas, 'attention_mask': masks, 'filenames': filenames, 'gold_images': gold_images}
    
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
            example_out['images'] = example['images']
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

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, \
            shuffle=True, collate_fn=collate_fn, worker_init_fn=np.random.seed(0), \
            num_workers=args.num_dataloader_workers)

    # Create and load models
    text_encoder = AutoModel.from_pretrained(args.encoder_model_type).cuda()
    # forward a fake batch to figure out cross_attention_dim
    hidden_states = encode_text(text_encoder, torch.zeros(1,1).long().cuda(), None)
    cross_attention_dim = hidden_states.shape[-1]
   
    image_decoder = create_image_decoder(image_size=args.image_size, color_channels=args.color_channels, \
            cross_attention_dim=cross_attention_dim)
    image_decoder = image_decoder.cuda()

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
    # Optimization
    optimizer = torch.optim.AdamW(image_decoder.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    train(train_dataloader, args.save_dir, args.save_model_every, \
        text_encoder, image_decoder, noise_scheduler, \
        args.scheduled_sampling_weights_start, args.scheduled_sampling_weights_end, \
        args.min_scheduled_sampling_step, \
        optimizer, lr_scheduler, args.num_epochs, \
        gradient_accumulation_steps=args.gradient_accumulation_steps, \
        clip_grad_norm=args.clip_grad_norm, \
        mixed_precision=args.mixed_precision)

    # Save final model
    save_model(image_decoder, os.path.join(args.save_dir, f'model_e{args.num_epochs}_lr{args.learning_rate}.pt.{args.num_epochs}'))


if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    main(args)

import math
import os
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import T5ForConditionalGeneration
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.hub_utils import init_git_repo, push_to_hub
from accelerate import notebook_launcher
from accelerate import Accelerator
from PIL import Image
torch.backends.cuda.matmul.allow_tf32=True
from tqdm.auto import tqdm
batch_size = 16
@dataclass
class TrainingConfig:
    image_size = (64, 320)  # the generated image resolution
    train_batch_size = batch_size
    eval_batch_size = batch_size  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    #learning_rate = 1e-4
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'latex2im'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()


config.dataset_name = "yuntian-deng/im2latex-100k"
dataset = load_dataset(config.dataset_name, split="val")

model_type = "EleutherAI/gpt-neo-125M"

#tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained(model_type, max_length=512)
#tokenizer.add_special_tokens({'pad_token': 'Ġgazed'})
#import pdb; pdb.set_trace()

preprocess = transforms.Compose(
    [
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Resize((config.image_size, config.image_size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def preprocess_formula(formula):
  example = tokenizer(formula)
  input_ids = example['input_ids']
  attention_mask = example['attention_mask']
  return input_ids, attention_mask

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    formulas_and_masks = [preprocess_formula(formula) for formula in examples['formula']]
    formulas = [item[0] for item in formulas_and_masks]
    masks = [item[1] for item in formulas_and_masks]
    return {"images": images, 'input_ids': formulas, 'attention_mask': masks}
    #return {"images": images}

dataset.set_transform(transform)
#import pdb; pdb.set_trace()
eos_id = tokenizer.encode(tokenizer.eos_token)[0]
def collate_fn(examples):
    #import pdb; pdb.set_trace()
    max_len = max([len(example['input_ids']) for example in examples]) + 1
    for example in examples:
        orig_len = len(example['input_ids'])
        formula = example['input_ids'] + [eos_id,] * (max_len - orig_len)
        example['input_ids'] = torch.LongTensor(formula)
        attention_mask = example['attention_mask'] + [1,] + [0,] * (max_len - orig_len - 1)
        example['attention_mask'] = torch.LongTensor(attention_mask)
    batch = default_collate(examples)
    #for k in batch:
    #    v = batch[k]
    #    if k != 'images':
    #        import pdb; pdb.set_trace()
    #        batch[k] = torch.LongTensor(v)
    return batch

torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)
#train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn, worker_init_fn=np.random.seed(0), num_workers=0)
eval_dataloader = train_dataloader




model = UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D", 
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D", 
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
            #"CrossAttnDownBlock2D", 
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
        ), 
        up_block_types=(
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",  # a regular ResNet upsampling block
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D", 
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D", 
            #"CrossAttnUpBlock2D", 
            "UpBlock2D",
            "UpBlock2D" 
          ),
          cross_attention_dim=768,
          mid_block_type='UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition'
    )



#t5_encoder = T5ForConditionalGeneration.from_pretrained('t5-small').encoder
t5_encoder = AutoModel.from_pretrained(model_type).cuda()
def forward_t5(input_ids, attention_mask):
  with torch.no_grad():
    #import pdb; pdb.set_trace()
    outputs = t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state 
    last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
  return last_hidden_state
    #mean = (masks.unsqueeze(-1) * last_hidden_state).sum(dim=-2) / masks.sum(-1, keepdim=True)



noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)





def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('L', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    for step, batch in enumerate(train_dataloader):
        clean_images = batch['images']
        input_ids = batch['input_ids'].cuda()
        masks = batch['attention_mask'].cuda()
        encoder_hidden_states = forward_t5(input_ids, masks)
        for iii, input_id in enumerate(input_ids):
            formula = tokenizer.decode(input_id, skip_special_symbols=True).replace('<|endoftext|>', '')
            print (f'{iii:04d}: {formula}')
            print ()
        images = pipeline(
            batch_size = config.eval_batch_size, 
            generator=torch.manual_seed(config.seed),
            encoder_hidden_states = encoder_hidden_states,
            attention_mask=masks,
        )["sample"]

    # Make a grid out of the images
        image_grid = make_grid(images, rows=batch_size, cols=1)

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{step:04d}.png")
        print ('='*10)
        break



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = init_git_repo(config, at_init=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            input_ids = batch['input_ids'].cuda()
            masks = batch['attention_mask'].cuda()
            encoder_hidden_states = forward_t5(input_ids, masks)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            break

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    push_to_hub(config, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir) 

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

#notebook_launcher(train_loop, args, num_processes=1)
state_dict = torch.load('model_e500_lr0.0001.pt.25', map_location='cpu')
state_dict_out = {}
for k in state_dict:
    k_out = k.replace('module.', '')
    state_dict_out[k_out] = state_dict[k]
model.load_state_dict(state_dict_out)
torch.manual_seed(1234)

accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
repo = init_git_repo(config, at_init=True)
push_to_hub(config, pipeline, repo, commit_message=f"init", blocking=True)

#evaluate(config, 0, pipeline)



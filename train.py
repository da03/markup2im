import math
import random
#import numpy as np
import os
from dataclasses import dataclass
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
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
#torch.backends.cuda.matmul.allow_tf32=True
from tqdm.auto import tqdm
random.seed(1234)
@dataclass
class TrainingConfig:
    image_size = (64, 320)  # the generated image resolution
    train_batch_size = 6
    eval_batch_size = 6  # how many images to sample during evaluation
    num_epochs = 500
    min_shuffle_steps = 100
    gradient_accumulation_steps = 1
    ss_probs_start = [0.1, 0.05, 0]
    ss_probs_end = [0.1, 0.2, 0.3]
    #ss_prob_end = 0.6
    #ss_prob_start = 0.1
    #learning_rate = 1e-4
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    #mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm-butterflies-128'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

#import pdb; pdb.set_trace()
config.dataset_name1 = "yuntian-deng/im2latex-100k-raw"
dataset1 = load_dataset(config.dataset_name1, split="train")
config.dataset_name2 = "yuntian-deng/im2latex-animals-200k"
dataset2 = load_dataset(config.dataset_name2, split="train[0:5000]") # use 5k images from animals
config.dataset_name3 = "yuntian-deng/im2latex-animals-concat"
dataset3 = load_dataset(config.dataset_name3, split="train[0:15000]") # use 15k images from animals
dataset = concatenate_datasets([dataset1, dataset2, dataset3])
#dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=42)

model_type = "EleutherAI/gpt-neo-125M"

#tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained(model_type, max_length=1024)
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
  example = tokenizer(formula, truncation=True, max_length=1024)
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
    return batch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)




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
          #cross_attention_dim=2048,
          cross_attention_dim=768,
          mid_block_type='UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition'
    )



    #mean = (masks.unsqueeze(-1) * last_hidden_state).sum(dim=-2) / masks.sum(-1, keepdim=True)

###state_dict = torch.load('model_e500_lr0.0001.pt.80', map_location='cpu')
###state_dict_out = {}
###for k in state_dict:
###    k_out = k.replace('module.', '')
###    state_dict_out[k_out] = state_dict[k]
###model.load_state_dict(state_dict_out)
#model.load_state_dict(torch.load('model_e500_lr0.0001.pt.80'))

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)





def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    )["sample"]

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


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
    #t5_encoder = T5ForConditionalGeneration.from_pretrained('t5-small').encoder
    t5_encoder = AutoModel.from_pretrained(model_type).to(accelerator.device)
    def forward_t5(input_ids, attention_mask):
      with torch.no_grad():
        #import pdb; pdb.set_trace()
        outputs = t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
        last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
      return last_hidden_state
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        ss_probs = []
        for ss_prob_start, ss_prob_end in zip(config.ss_probs_start, config.ss_probs_end):
            ss_prob = ss_prob_start + epoch / config.num_epochs * (ss_prob_end - ss_prob_start)
            ss_probs.append(ss_prob)

        #import pdb; pdb.set_trace()
        ss_probs.insert(0, 1-sum(ss_probs))
        acc_probs = []
        acc_prob = 0
        for p in ss_probs:
            acc_prob += p
            acc_probs.append(acc_prob)
        print ('='*10)
        print (f'ss probs: {ss_probs}, acc probs: {acc_probs}')
        print ('='*10)
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(accelerator.device)
            input_ids = batch['input_ids'].to(accelerator.device)
            masks = batch['attention_mask'].to(accelerator.device)
            encoder_hidden_states = forward_t5(input_ids, masks)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
            #import pdb; pdb.set_trace()
            p = random.random()
            for num_ss_steps in range(len(ss_probs)):
                acc_prob = acc_probs[num_ss_steps]
                if p < acc_prob:
                    break

            #num_ss_steps = np.random.choice(list(range(len(ss_probs))), p=ss_probs)
            flag_ss = num_ss_steps > 0
            if max(timesteps) >= noise_scheduler.num_train_timesteps-num_ss_steps:
                flag_ss = False
            if flag_ss:
                with torch.no_grad():
                    #import pdb; pdb.set_trace()
                    # first, sample t + num_ss_steps
                    noise1 = noise
                    noise = None
                    noisy_images_t_plus_s = noise_scheduler.add_noise(clean_images, noise1, timesteps+num_ss_steps)
                    # next, decode and clean
                    for s in range(num_ss_steps):
                        # predict noise
                        noise_pred_t_plus_s = model(noisy_images_t_plus_s, timesteps+num_ss_steps-s, encoder_hidden_states, attention_mask=masks)["sample"]
                        lambs_t_plus_s, alpha_prod_ts_t_plus_s = noise_scheduler.get_lambda_and_alpha(timesteps+num_ss_steps-s)
                        # clean img predicted
                        x_0_pred = (noisy_images_t_plus_s - lambs_t_plus_s.view(-1, 1, 1, 1) * noise_pred_t_plus_s) / alpha_prod_ts_t_plus_s.view(-1, 1, 1, 1)
                        noise2 = torch.randn(clean_images.shape).to(clean_images.device)
                        # get previous step sample
                        noisy_images_t_plus_s_minus_one  = noise_scheduler.add_noise(x_0_pred, noise2, timesteps + num_ss_steps-s-1)
                        # update
                        noisy_images_t_plus_s = noisy_images_t_plus_s_minus_one
                    noisy_images_t = noisy_images_t_plus_s
            else:
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                if not flag_ss:
                    noise_pred = model(noisy_images, timesteps, encoder_hidden_states, attention_mask=masks)["sample"]
                else:
                    #import pdb; pdb.set_trace()
                    lambs_t, alpha_prod_ts_t = noise_scheduler.get_lambda_and_alpha(timesteps)
                    noise_pred = model(noisy_images_t, timesteps, encoder_hidden_states, attention_mask=masks)["sample"]
                    noise = (noisy_images_t - alpha_prod_ts_t.view(-1, 1, 1, 1) * clean_images) / lambs_t.view(-1, 1, 1, 1)
                    #noise_2 = noise2 + (lambs_t_plus_one/lambs_t).view(-1, 1, 1, 1) * (noise1 - noise_pred_t_plus_one) * ( alpha_prod_ts_t / alpha_prod_ts_t_plus_one).view(-1, 1, 1, 1)
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
            #break
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'model_e{config.num_epochs}_lr{config.learning_rate}.pt.{epoch}')

        ## After each epoch you optionally sample some demo images with evaluate() and save the model
        #if accelerator.is_main_process:
        #    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

        #    if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        #        evaluate(config, epoch, pipeline)

        #    if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        #        if config.push_to_hub:
        #            push_to_hub(config, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=True)
        #        else:
        #            pipeline.save_pretrained(config.output_dir) 

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

train_loop(*args)
#notebook_launcher(train_loop, args, num_processes=1)

torch.save(model.state_dict(), f'model_e{config.num_epochs}_lr{config.learning_rate}.pt')

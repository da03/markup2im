import math
import os
import numpy as np
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
from PIL import Image
#torch.backends.cuda.matmul.allow_tf32=True
import tqdm
#batch_size = 32
batch_size = 16
@dataclass
class TrainingConfig:
    image_size = (64, 320)  # the generated image resolution
    eval_batch_size = batch_size  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    #learning_rate = 1e-4
    learning_rate = 3e-5
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'images_rendered'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

config.dataset_name1 = "yuntian-deng/im2latex-100k-raw"
dataset1 = load_dataset(config.dataset_name1, split="val")
config.dataset_name2 = "yuntian-deng/im2latex-animals-200k"
dataset2 = load_dataset(config.dataset_name2, split="val") # use 15k images from animals
config.dataset_name3 = "yuntian-deng/im2latex-animals-concat"
dataset3 = load_dataset(config.dataset_name3, split="val") # use 15k images from animals
dataset = concatenate_datasets([dataset1, dataset2, dataset3])
#dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=42)


# for actual testing
config.dataset_name = "yuntian-deng/im2latex-100k"
dataset = load_dataset(config.dataset_name, split="test")
dataset = dataset.shuffle(seed=42)

#config.dataset_name = "yuntian-deng/im2latex-100k-raw"
#dataset = load_dataset(config.dataset_name, split="val")

model_type = "EleutherAI/gpt-neo-125M"

#tokenizer = AutoTokenizer.from_pretrained("text-small")
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
  #formula1 = 'a + b = c'
  #formula2 = 'a+b=c'
  #formula3 = r'd s _ { 1 1 } ^ { 2 } = d x ^ { + } d x ^ { - } + l _ { p } ^ { 9 } \frac { p _ { - } } { r ^ { 7 } } \delta ( x ^ { - } ) d x ^ { - } d x ^ { - }'
  #formula4 = '1 + 2 = 3'
  #formula5 = r'\cat \elephant \frog'
  #formula6 = r'1 + \cat - 2 + \elephant = \frog'
  #formula7 = r'a + b \cat - c + \elephant = \frog'
  #formula8 = r'd s _ { 1 1 } ^ { 2 } \frog = \crab d x ^ { + } d x ^ { - } + l _ { p } ^ { 9 } \frac { p _ { - } } { r ^ { 7 } } \delta ( x ^ { - } ) d x ^ { - } d x ^ { - }'
  #import random
  #formula = random.choice([formula1, formula2, formula3, formula4, formula5, formula6, formula7, formula8])
  example = tokenizer(formula, truncation=True, max_length=1024)
  input_ids = example['input_ids']
  attention_mask = example['attention_mask']
  return input_ids, attention_mask

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    gold_images = [image for image in examples["image"]]
    formulas_and_masks = [preprocess_formula(formula) for formula in examples['formula']]
    formulas = [item[0] for item in formulas_and_masks]
    masks = [item[1] for item in formulas_and_masks]
    filenames = examples['filename']
    return {"images": images, 'input_ids': formulas, 'attention_mask': masks, 'filenames': filenames, 'gold_images': gold_images}
    #return {"images": images}

dataset.set_transform(transform)
#import pdb; pdb.set_trace()
eos_id = tokenizer.encode(tokenizer.eos_token)[0]
def collate_fn(examples):
    #import pdb; pdb.set_trace()
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
#train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn, worker_init_fn=np.random.seed(0), num_workers=0)
eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn, worker_init_fn=np.random.seed(0), num_workers=0)




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

model = model.cuda()


#text_encoder = T5ForConditionalGeneration.from_pretrained('text-small').encoder
text_encoder = AutoModel.from_pretrained(model_type).cuda()
def forward_text(input_ids, attention_mask):
  with torch.no_grad():
    #import pdb; pdb.set_trace()
    outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state 
    last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
  return last_hidden_state
    #mean = (masks.unsqueeze(-1) * last_hidden_state).sum(dim=-2) / masks.sum(-1, keepdim=True)



noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
#
#optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
#
#
#lr_scheduler = get_cosine_schedule_with_warmup(
#    optimizer=optimizer,
#    num_warmup_steps=config.lr_warmup_steps,
#    num_training_steps=(len(train_dataloader) * config.num_epochs),
#)





def make_grid(images, rows, cols):
    w, h = images[0].size
    #import pdb; pdb.set_trace()
    if images[0].mode != 'RGB':
        grid = Image.new('L', size=(cols*w, rows*h))
    else:
        grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # Save the images
    gold_dir = os.path.join(config.output_dir, "images_gold")
    os.makedirs(gold_dir, exist_ok=True)
    pred_dir = os.path.join(config.output_dir, "images_pred")
    os.makedirs(pred_dir, exist_ok=True)
    for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
        gold_images = batch['gold_images']
        filenames = batch['filenames']
        clean_images = batch['images']
        input_ids = batch['input_ids'].cuda()
        masks = batch['attention_mask'].cuda()
        encoder_hidden_states = forward_text(input_ids, masks)
        for iii, input_id in enumerate(input_ids):
            formula = tokenizer.decode(input_id, skip_special_symbols=True).replace('<|endoftext|>', '')
            print (f'{iii:04d}: {formula}')
            print ()
        swap_step = -1
        pred_images = pipeline.swap(
            batch_size = config.eval_batch_size, 
            generator=torch.manual_seed(config.seed),
            encoder_hidden_states = encoder_hidden_states,
            attention_mask=masks,
            swap_step=swap_step,
        )["sample"]

        for filename, gold_image, pred_image in zip(filenames, gold_images, pred_images):
            gold_image.save(os.path.join(gold_dir, filename))
            pred_image.save(os.path.join(pred_dir, filename))
        #break

        # Make a grid out of the images
        #image_grid = make_grid(images, rows=batch_size, cols=1)

        #image_grid.save(f"{test_dir}/swap_{swap_step}.png")
        print ('='*10)
        break



state_dict = torch.load('model_e500_lr0.0001.pt.45', map_location='cpu')
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
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

evaluate(config, 0, pipeline)



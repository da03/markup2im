import gradio as gr
import torch

from PIL import Image
import numpy as np
from io import BytesIO
import os

from diffusers import DDPMPipeline
from transformers import AutoTokenizer, AutoModel

import diffusers
print (diffusers.__file__)

# setup
def setup(device='cpu'):
    img_pipe = DDPMPipeline.from_pretrained("yuntian-deng/latex2im")
    img_pipe.to(device)
    
    model_type = "EleutherAI/gpt-neo-125M"
    encoder = AutoModel.from_pretrained(model_type).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_type, max_length=512)
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    
    def forward_encoder(latex):
        encoded = tokenizer(latex, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encoded['input_ids']
        input_ids = torch.cat((input_ids, torch.LongTensor([eos_id,]).unsqueeze(0)), dim=-1)
        input_ids = input_ids.to(device)
        attention_mask = encoded['attention_mask']
        attention_mask = torch.cat((attention_mask, torch.LongTensor([1,]).unsqueeze(0)), dim=-1)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state 
            last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state # shouldn't be necessary
        return last_hidden_state
    return img_pipe, forward_encoder

img_pipe, forward_encoder = setup()
gallery = gr.Gallery(label="Rendered Image", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

# infer
def infer(latex): 
    encoder_hidden_states = forward_encoder(latex)
    i = 0
    for image, image_clean in img_pipe.run_clean(batch_size=1, generator=torch.manual_seed(0), encoder_hidden_states=encoder_hidden_states, output_type="numpy"):
        i += 1
        yield i, [image[0], image_clean[0]]

title="Diffusion Based Latex Renderer 8gpus ss e40"
description="Warning: Slow process... ~20 min inference time."

# launch
gr.Interface(fn=infer, inputs=["text"], outputs=[gr.Slider(0, 1000, value=0, label='step'), gallery],title=title,description=description).queue(max_size=100).launch(enable_queue=True)

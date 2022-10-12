import os
import torch
from diffusers import UNet2DConditionModel

def create_image_decoder(image_size, color_channels, cross_attention_dim):
    image_decoder = UNet2DConditionModel(
        sample_size=image_size,
        in_channels=color_channels,
        out_channels=color_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D", 
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
        ), 
        up_block_types=(
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "UpBlock2D",
            "UpBlock2D" 
          ),
          cross_attention_dim=cross_attention_dim,
          mid_block_type='UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition'
    )
    return image_decoder

def encode_text(text_encoder, input_ids, attention_mask, no_grad=True):
    if no_grad:
        with torch.no_grad():
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state 
            if attention_mask is not None:
                last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
    else:
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
        if attention_mask is not None:
            last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
    return last_hidden_state

def save_model(model, filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), filename)

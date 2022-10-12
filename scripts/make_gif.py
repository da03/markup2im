import glob
import random
import sys
import os
import tqdm
import argparse
import numpy as np
from PIL import Image

def process_args(args):
    parser = argparse.ArgumentParser(description='Make gif from diffusion intermediate images.')

    parser.add_argument('--input_dir',
                        type=str, required=True,
                        help=('Input directory containing images_pred and images_gold.'
                        ))
    parser.add_argument('--filter_filename',
                        type=str, required=True,
                        help=('Filename for which to generate a gif.'
                        ))
    parser.add_argument('--output_filename',
                        type=str, required=True,
                        help=('Output filename.'
                        ))
    parser.add_argument('--show_every',
                        type=int, default=-1,
                        help=('Saves intermediate diffusion steps every this many steps. When <0 does not save any intermediate images.'
                        ))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    filenames = glob.glob(os.path.join(args.input_dir, f'images_pred/{args.filter_filename}_*'))
    assert len(filenames) > 0, f'Please make sure that the input folder contains images with prefix {args.filter_filename}'
    #import pdb; pdb.set_trace()
    filenames = sorted(list(filenames))
    frames = []
    for i, filename in enumerate(filenames):
        if args.show_every < 0 or i % args.show_every == 0:
            new_frame = Image.open(filename)
            frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save(args.output_filename, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=1, loop=0)



if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    main(args)

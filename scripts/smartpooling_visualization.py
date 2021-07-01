import argparse
import json
import os
import numpy as np
import torch
import time
from copy import deepcopy
import random
from PIL import Image, ImageDraw, ImageFont
import psutil
import sys

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    parser.add_argument('--pathToCapture', type=str, default=None,
                          help='Path to the directory containing the '
                          'captured data.')
    parser.add_argument('--smartpoolingInAR', action='store_true',
                       help='Put smart averaging in AR. So archtecture is encoder -> (smart averaging -> AR) instead of (encoder -> smart averaging) -> AR')  

    args = parser.parse_args(argv)

    return args

def main(args):
    args = parseArgs(args)

    dir_name = "smartpooling_importance" + ("_ar" if args.smartpoolingInAR else "")
    full_dir_name = os.path.join(args.pathToCapture, dir_name)
    filename = sorted([f for f in os.listdir(full_dir_name) if os.path.isfile(os.path.join(full_dir_name, f)) and f.startswith(dir_name)])[0]
    filename_suffix = filename.split(dir_name)[1]

    batchData = torch.load(os.path.join(full_dir_name, "batchData" + filename_suffix))
    importance = torch.load(os.path.join(full_dir_name, dir_name + filename_suffix))
    phone_align = torch.load(os.path.join(args.pathToCapture, "phone_align", "phone_align" + filename_suffix))
    phone_to_id = torch.load(os.path.join(os.path.dirname(args.pathToCapture), "labelsToIdDict.pt"))
    id_to_phone = {phone_id : phone for phone, phone_id in phone_to_id.items()}

    half_size = 16
    text_offset = 3
    DOWNSAMPLING = 160
    x_axis_zoom_out = 16
    lines_to_visualize = min(16, importance.shape[0])

    batchData = batchData[:lines_to_visualize]
    importance = importance[:lines_to_visualize]
    phone_align = phone_align[:lines_to_visualize]

    def draw_text(draw, x, y, text, font):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x+dx, y+dy), text, font=font, fill='black', align="left")
        draw.text((x, y), text, font=font, fill='white', align="left")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    with torch.no_grad():        
        importance = np.array(importance.numpy()*255, dtype=np.int32).repeat(int(batchData.shape[2]/importance.shape[1]/x_axis_zoom_out), axis=1)
        importance = importance.repeat(2*half_size, axis=0)
        img = Image.fromarray(importance).convert('RGB')

        draw = ImageDraw.Draw(img)
        for b in range(batchData.shape[0]):
            for i in range(batchData.shape[2]-1):
                draw.line([(i / x_axis_zoom_out, round(half_size * (2*b + 1 + batchData[b, 0, i].item()))), ((i+1) / x_axis_zoom_out, round(half_size * (2*b + 1 + batchData[b, 0, i+1].item())))], fill='blue', width=1)
        
        for b in range(phone_align.shape[0]):
            last = -1
            for i in range(phone_align.shape[1]):
                if last != phone_align[b,i]:
                    draw_text(draw, i * DOWNSAMPLING / x_axis_zoom_out + text_offset, 2 * b * half_size + text_offset, id_to_phone[phone_align[b,i].item()], font)
                    draw.line([(i * DOWNSAMPLING / x_axis_zoom_out, 2 * b * half_size), (i * DOWNSAMPLING / x_axis_zoom_out, 2 * (b + 1) * half_size)], fill='red', width=1)
                last = phone_align[b,i]

        img.save(os.path.join(args.pathToCapture, "smartpooling_importance.png"))

if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()

    args = sys.argv[1:]
    main(args)
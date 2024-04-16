import numpy as np
import re
from PIL import Image
import sys
import os
import json
import torch
from glob import glob
import logging


def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img



def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

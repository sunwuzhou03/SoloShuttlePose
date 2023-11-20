import cv2
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os

import warnings
import torch
import torchvision
from tqdm import tqdm
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

sys.path.append("src/models")
sys.path.append("src/tools")

from TrackNet import TrackNet
from utils import extract_numbers, write_json, read_json,clear_file
from denoise import smooth
from event_detection import event_detect
# clear the polyfit Rankwarning
warnings.simplefilter('ignore', np.RankWarning)

result_path="res"
name_list=[
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "test6"
]
for video_name in name_list:
    clear_file(video_name,f"{result_path}")
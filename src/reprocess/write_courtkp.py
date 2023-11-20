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


folder_path = "E:/SoloShuttlePoseRes/new_res/courts/court_kp"#"references"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in ['.json']:
            refer_json_path = os.path.join(root, file)

            print(refer_json_path)

            courtkp_json_path=os.path.join(root, file)
            
            refer_dict=read_json(refer_json_path)
            courtkp_dict=read_json(courtkp_json_path)

            json_name = os.path.basename(refer_json_path).split('.')[0]

            normal_court_info = refer_dict['court_info']
            normal_net_info = refer_dict['net_info']
            first_frame=courtkp_dict['first_rally_frame']
            next_frame=courtkp_dict['next_rally_frame']

            # correct net position
            if normal_net_info is not None:
                if normal_court_info is not None:
                    normal_net_info[1][1],normal_net_info[2][1]=\
                        normal_court_info[2][1],normal_court_info[3][1]

            court_dict = {
                "first_rally_frame": first_frame,
                "next_rally_frame": next_frame,
                "court_info": normal_court_info,
                "net_info": normal_net_info,
            }

            write_json(court_dict, json_name,
                       f"E:/ans/tem/courts/court_kp", "w")
            
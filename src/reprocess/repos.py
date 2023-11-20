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

# This program just to use to correct the old version results

result_path="E:\\res"
name_list=[
    "Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals",
    "Akane_YAMAGUCHI_AN_Se_Young_DAIHATSU_YONEX_Japan_Open_2022_Finals",
    "Akane_YAMAGUCHI_CHEN_Yu_Fei_BWF_World_Championships_2022_Finals",
    "Akane_YAMAGUCHI_CHEN_Yu_Fei_England_Open_2022_Semi_finals",
    "An_Se_Young_Chen_Yu_Fei_PERODUA_Malaysia_Masters_2022_Finals",
    "AN_Se_Young_Gregoria_Mariska_TUNJUNG_Malaysia_Masters_2022_SemiFinals",
    "AN_Se_Young_Pornpawee_CHOCHUWONG_Korea_Open_Badminton_Championships_2022_Finals",
    "AN_Seyoung_PUSARLA_V__Sindhu_Korea_Open_Badminton_Championships_2022_Semi_Final",
    "Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals",
    "Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals",
]
print(result_path)
for video_name in name_list:
    print("+" * 10 + "Starting Ball Detection" + "+" * 10)       
    json_path=os.path.join(f"{result_path}/players/player_kp",f"{video_name}.json")
    print(json_path)
    # break

    info_dict=read_json(json_path)
    player_dict={}
    # 遍历 JSON 对象中的每个键值对
    for key, value in info_dict.items():
        # print(key)
        player_dict[key]={
            'top':value['bottom'],
            'bottom':value['top']
        }

        write_json(player_dict, video_name,f"E:/ans/res/players/player_kp")
        player_dict={}

    print("+" * 10 + "End Badminton Detection" + "+" * 10)

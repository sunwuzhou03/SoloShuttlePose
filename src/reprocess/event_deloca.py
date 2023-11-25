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


# name_list=[
#     "Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals",
#     "Akane_YAMAGUCHI_AN_Se_Young_DAIHATSU_YONEX_Japan_Open_2022_Finals",
#     "Akane_YAMAGUCHI_CHEN_Yu_Fei_BWF_World_Championships_2022_Finals",
#     "Akane_YAMAGUCHI_CHEN_Yu_Fei_England_Open_2022_Semi_finals",
#     "An_Se_Young_Chen_Yu_Fei_PERODUA_Malaysia_Masters_2022_Finals",
#     "AN_Se_Young_Gregoria_Mariska_TUNJUNG_Malaysia_Masters_2022_SemiFinals",
#     "AN_Se_Young_Pornpawee_CHOCHUWONG_Korea_Open_Badminton_Championships_2022_Finals",
#     "AN_Seyoung_PUSARLA_V",
#     "Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals",
#     "Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals",
# ]
# result_path="res"
# name_list=[
#     "test1",
#     "test2",
#     "test3",
#     "test4",
#     "test5",
#     "test6"
# ]
# for video_name in name_list:
#     clear_file(video_name,f"{result_path}")

# for video_name in name_list:
    
#     clear_file(video_name,f"{result_path}/ball/loca_info(denoise)")
#     clear_file(video_name,f"{result_path}/ball/event")
#     clear_file(video_name,f"{result_path}/ball/traj2img")

result_path="E:/SoloShuttlePoseRes/res"

clear_file("loca_info(denoise)",f"{result_path}/ball")
clear_file("event",f"{result_path}/ball")
clear_file("traj2img",f"{result_path}/ball")

print("-" * 10 + "Starting Ball Detection" + "" * 10)       
for res_root, res_dirs, res_files in os.walk(f"{result_path}/ball/loca_info"):
    for res_file in res_files:
        _, ext = os.path.splitext(res_file)
 
        if ext.lower() in ['.json']:

            # fake video
            video_path = os.path.join(res_root, res_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            orivi_name, start_frame = extract_numbers(video_name)    
            print(video_name)      
            # denoise file save path
            dd_save_dir = os.path.join(f"{result_path}/ball", f"loca_info(denoise)/{orivi_name}")
            os.makedirs(dd_save_dir, exist_ok=True)

            d_save_dir = os.path.join(f"{result_path}/ball", f"loca_info/{orivi_name}")
            json_path = f"{d_save_dir}/{video_name}.json"

            cd_save_dir= os.path.join(f"{result_path}/courts", f"court_kp")
            cd_json_path=f"{cd_save_dir}/{orivi_name}.json"
            court=read_json(cd_json_path)['court_info']            
            
            smooth(json_path,court ,dd_save_dir)
            
            dd_json_path = f"{dd_save_dir}/{video_name}.json"
            # event_detect(dd_json_path, f"{result_path}/ball")
print("" * 10 + "End Badminton Detection" + "" * 10)

import cv2
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os
import sys
sys.path.append("src/models")
sys.path.append("src/tools")

from utils import write_json, read_json,clear_file, is_video_detect, find_next, find_reference
from VideoClip import VideoClip
from PoseDetect import PoseDetect
from CourtDetect import CourtDetect
from NetDetect import NetDetect
import argparse
from BallDetect import ball_detect
import warnings
# clear the polyfit Rankwarning
warnings.simplefilter('ignore', np.RankWarning)

players_dict = read_json("E:\SoloShuttlePoseRes/new_res/players/player_kp//He_Bing_Jiao_Chen_Yu_Fei_GWANGJU_YONEX_Korea_Masters_2022_Final.json")

# you can set loca_info which not to denoise
ball_dict=read_json("E:\SoloShuttlePoseRes/new_res/ball/loca_info(denoise)/He_Bing_Jiao_Chen_Yu_Fei_GWANGJU_YONEX_Korea_Masters_2022_Final/He_Bing_Jiao_Chen_Yu_Fei_GWANGJU_YONEX_Korea_Masters_2022_Final_107710-108326.json")
court_dict=read_json("E:/SoloShuttlePoseRes/new_res/courts/court_kp/He_Bing_Jiao_Chen_Yu_Fei_GWANGJU_YONEX_Korea_Masters_2022_Final.json")


image_list=[]
fkey=-1
for key,value in ball_dict.items():
    if fkey==-1:
        fkey=int(key)
    current_frame=int(key)

    # figure 2
    plt.figure(figsize=(16, 12)) 
    plt.ylim(0, 1080)
    plt.xlim(0, 1920)
    plt.gca().invert_yaxis()
    
    # 给定的点
    fn=current_frame
    joints = players_dict[f"{fn}"]
    while fn>fkey and (joints['top'] is None or joints['bottom'] is None):
        fn-=1
        joints = players_dict[f"{fn}"]
    
    if fn==fkey:
        continue
    
    
    ball=(ball_dict[str(current_frame)]['x'],ball_dict[str(current_frame)]['y'])
    
    players_joints = joints['top']


    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in players_joints]
    y = [joint[1] for joint in players_joints]


    # 创建散点图
    plt.scatter(x, y,c="b")



    edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)]

    # 循环添加标号
    for i, joint in enumerate(players_joints):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'b-')

    players_joints = joints['bottom']

    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in players_joints]
    y = [joint[1] for joint in players_joints]

    # 创建散点图
    plt.scatter(x, y,c="r")

    edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)]

    # 循环添加标号
    for i, joint in enumerate(players_joints):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'r-')


    # 场
    court=court_dict['court_info']
    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in court]
    y = [joint[1] for joint in court]
    # 创建散点图
    plt.scatter(x, y,c="y")
    edges = [(0, 1), (2, 3), (4, 5),(0,4),(1,5)]
    # 循环添加标号
    for i, joint in enumerate(court):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')


    # 网
    net=court_dict['net_info']
    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in net]
    y = [joint[1] for joint in net]
    # 创建散点图
    plt.scatter(x, y,c="y")
    edges = [(0, 1), (1, 2), (2, 3),(0,3)]
    # 循环添加标号
    for i, joint in enumerate(net):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')
    
    # 球
   
    if ball_dict[str(current_frame)]['visible']!=0 :    
        plt.scatter(ball[0], ball[1],c="r")
        plt.annotate("ball", (ball[0], ball[1]), textcoords="offset points", xytext=(0,10), ha='center')



    # 设置图形标题和轴标签
    plt.title('All_info')
    plt.xlabel('X')
    plt.ylabel('Y')

    from PIL import Image
    import copy

    # 使用PIL库加载图像文件，并将其添加到图像列表中
    plt.savefig('Frame.png')
    image_list.append(Image.open('Frame.png').copy())
    
    plt.clf()
    plt.close()

os.remove("Frame.png")
# 保存为GIF文件
image_list[0].save('Dynamic_diagram_game-cpt.gif', save_all=True, append_images=image_list[1:], duration=66, loop=0)

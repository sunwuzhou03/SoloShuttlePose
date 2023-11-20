import cv2
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os
import sys
import argparse
from collections import deque
from PIL import Image, ImageDraw

sys.path.append("src/tools")
sys.path.append("src/models")

from utils import write_json, clear_file, is_video_detect, find_next, find_reference, read_json
from VideoClip import VideoClip
from PoseDetect import PoseDetect
from CourtDetect import CourtDetect
from NetDetect import NetDetect

parser = argparse.ArgumentParser(description='para transfer')
parser.add_argument('--folder_path',
                    type=str,
                    default="videos",
                    help='folder_path -> str type.')
parser.add_argument('--result_path',
                    type=str,
                    default="res",
                    help='result_path -> str type.')
parser.add_argument('--force',
                    action='store_true',
                    default=False,
                    help='force -> bool type.')
parser.add_argument('--court',
                    action='store_true',
                    default=False,
                    help='court -> bool type.')
parser.add_argument('--net',
                    action='store_true',
                    default=False,
                    help='net -> bool type.')
parser.add_argument('--players',
                    action='store_true',
                    default=False,
                    help='players -> bool type.')
parser.add_argument('--ball',
                    action='store_true',
                    default=False,
                    help='ball -> bool type.')
parser.add_argument('--trajectory',
                    action='store_true',
                    default=False,
                    help='trajectory -> bool type.')
parser.add_argument('--traj_len',
                    type=int,
                    default=8,
                    help='traj_len -> int type.')
args = parser.parse_args()
print(args)

folder_path = args.folder_path
force = args.force
result_path = args.result_path
court = args.court
net = args.net
players = args.players
ball = args.ball
trajectory = args.trajectory
traj_len = args.traj_len

for root, dirs, files in os.walk(folder_path):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in ['.mp4']:
            video_path = os.path.join(root, file)
            print(video_path)
            video_name = os.path.basename(video_path).split('.')[0]

            process_video_path = f"{result_path}/videos/{video_name}/{video_name}.mp4"
            if os.path.exists(process_video_path):
                if force:
                    os.remove(process_video_path)
                else:
                    continue

            full_video_path = os.path.join(f"{result_path}/videos", video_name)
            if not os.path.exists(full_video_path):
                os.makedirs(full_video_path)

            # read ball location
            ball_dict = {}
            for res_root, res_dirs, res_files in os.walk(
                    f"{result_path}/ball/loca_info(denoise)/{video_name}"):
                for res_file in res_files:
                    print(res_root)
                    _, ext = os.path.splitext(res_file)
                    if ext.lower() in ['.json']:
                        res_json_path = os.path.join(res_root, res_file)
                        ball_dict.update(read_json(res_json_path))

            # Open the video file
            video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Define the codec for the output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = f'{full_video_path}/{video_name}.mp4'
            video_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                           (width, height))

            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            reference_path = find_reference(video_name, "res/courts/court_kp")
            pose_detect = PoseDetect()
            court_detect = CourtDetect()
            net_detect = NetDetect()

            _ = court_detect.pre_process(video_path, reference_path)
            _ = net_detect.pre_process(video_path, reference_path)

            reference_path = find_reference(video_name,
                                            "res/players/player_kp")
            players_dict = read_json(reference_path)

            with tqdm(total=total_frames) as pbar:
                traj_queue = deque()
                while True:
                    # Read a frame from the video
                    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = video.read()
                    # If there are no more frames, break the loop
                    if not ret:
                        break

                    joints = players_dict[f"{current_frame}"]
                    players_joints = [joints['top'], joints['bottom']]

                    can_draw = True
                    if players_joints[0] is None or players_joints[1] is None:
                        can_draw = False
                    if can_draw:
                        if court:
                            # draw human, court, players
                            frame = court_detect.draw_court(frame)
                        if net:
                            frame = net_detect.draw_net(frame)
                        if players:
                            frame = pose_detect.draw_key_points(
                                players_joints, frame)
                        if ball:
                            if str(current_frame) in ball_dict:
                                loca_dict = ball_dict[f"{current_frame}"]
                                if loca_dict["visible"] == 1:
                                    x = int(loca_dict['x'])
                                    y = int(loca_dict['y'])
                                    cv2.circle(frame, (x, y), 8, (0, 0, 255),
                                               -1)
                        if trajectory:
                            if str(current_frame) in ball_dict:
                                loca_dict = ball_dict[f"{current_frame}"]
                                # Push ball coordinates for each frame
                                if loca_dict["visible"] == 1:
                                    x = int(loca_dict['x'])
                                    y = int(loca_dict['y'])
                                    if len(traj_queue) >= traj_len:
                                        traj_queue.pop()
                                    traj_queue.appendleft([x, y])
                                else:
                                    if len(traj_queue) >= traj_len:
                                        traj_queue.pop()
                                    traj_queue.appendleft(None)

                                # Convert to PIL image for drawing
                                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(img)

                                # Draw ball trajectory
                                for i in range(len(traj_queue)):
                                    if traj_queue[i] is not None:
                                        draw_x = traj_queue[i][0]
                                        draw_y = traj_queue[i][1]
                                        bbox = (draw_x - 2, draw_y - 2,
                                                draw_x + 2, draw_y + 2)
                                        draw = ImageDraw.Draw(img)
                                        draw.ellipse(bbox, outline='yellow')
                                        del draw

                                # Convert back to cv2 image and write to output video
                                frame = cv2.cvtColor(np.array(img),
                                                     cv2.COLOR_RGB2BGR)

                    video_writer.write(frame)
                    pbar.update(1)
            # Release the video capture and writer objects
            video.release()
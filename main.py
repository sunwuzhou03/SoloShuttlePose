import cv2
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os
from src.tools.utils import write_json, clear_file, is_video_detect, find_next, find_reference
from src.tools.VideoClip import VideoClip
from src.models.PoseDetect import PoseDetect
from src.models.CourtDetect import CourtDetect
from src.models.NetDetect import NetDetect
import argparse
from src.tools.BallDetect import ball_detect
import logging
import traceback
import warnings

# clear the polyfit Rankwarning
warnings.simplefilter('ignore', np.RankWarning)

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

args = parser.parse_args()
print(args)

folder_path = args.folder_path
force = args.force
result_path = args.result_path

for root, dirs, files in os.walk(folder_path):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in ['.mp4']:
            video_path = os.path.join(root, file)
            print(video_path)
            video_name = os.path.basename(video_path).split('.')[0]

            if is_video_detect(video_name):
                if force:
                    clear_file(video_name)
                else:
                    continue

            full_video_path = os.path.join(f"{result_path}/videos", video_name)
            if not os.path.exists(full_video_path):
                os.makedirs(full_video_path)

            # Open the video file
            video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # write video information
            video_dict = {
                "video_name": video_name,
                "fps": fps,
                "height": height,
                "width": width,
                "total_frames": total_frames
            }
            write_json(video_dict, video_name, full_video_path)

            # example class
            pose_detect = PoseDetect()
            court_detect = CourtDetect()
            net_detect = NetDetect()
            video_cilp = VideoClip(video_name, fps, total_frames, width,
                                   height, full_video_path)

            reference_path = find_reference(video_name)
            if reference_path is None:
                print(
                    "There is not reference frame! Now try to find it automatically. "
                )
            else:
                print(f"The reference frame is {reference_path}. ")

            # begin_frame is a rough estimate of valid frames
            begin_frame = court_detect.pre_process(video_path, reference_path)
            _ = net_detect.pre_process(video_path, reference_path)

            # next_frame is a more accurate estimate of the effective frame using bisection search
            next_frame = find_next(video_path, court_detect, begin_frame)
            first_frame = next_frame

            normal_court_info = court_detect.normal_court_info
            normal_net_info = net_detect.normal_net_info

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

            write_json(court_dict, video_name,
                       f"{result_path}/courts/court_kp", "w")

            # release memory
            # net_detect.del_RCNN()

            with tqdm(total=total_frames) as pbar:

                while True:
                    # Read a frame from the video
                    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = video.read()
                    # If there are no more frames, break the loop
                    if not ret:
                        break
                    # assume it don't detect anything
                    have_court = False
                    players_dict = {
                        str(current_frame): {
                            "top": None,
                            "bottom": None
                        }
                    }
                    have_court_dict = {str(current_frame): have_court}

                    final_frame = frame.copy()

                    if current_frame < next_frame:
                        write_json(have_court_dict, video_name,
                                   f"{result_path}/courts/have_court")
                        write_json(players_dict, video_name,
                                   f"{result_path}/players/player_kp")
                        court_mse_dict = {str(current_frame): court_detect.mse}
                        write_json(court_mse_dict, video_name,
                                   f"{result_path}/courts/court_mse")

                        video_made = video_cilp.add_frame(
                            have_court, frame, current_frame)

                        pbar.update(1)
                        continue

                    # player detect and court detect
                    court_info, have_court = court_detect.get_court_info(frame)
                    if have_court:

                        # pose_detect.setup_RCNN()
                        original_outputs, human_joints = pose_detect.get_human_joints(
                            frame)
                        # pose_detect.del_RCNN()

                        have_player, players_joints = court_detect.player_detection(
                            original_outputs)

                        if have_player:
                            players_dict = {
                                str(current_frame): {
                                    "top": players_joints[0],
                                    "bottom": players_joints[1]
                                }
                            }

                    video_made = video_cilp.add_frame(have_court, frame,
                                                      current_frame)
                    if video_made:
                        next_frame = find_next(video_path, court_detect,
                                               current_frame)
                        court_dict = {
                            "first_rally_frame": first_frame,
                            "next_rally_frame": next_frame,
                            "court_info": normal_court_info,
                            "net_info": normal_net_info,
                        }
                        write_json(court_dict, video_name,
                                   f"{result_path}/courts/court_kp", "w")

                    have_court_dict = {str(current_frame): True}
                    court_mse_dict = {str(current_frame): court_detect.mse}
                    write_json(court_mse_dict, video_name,
                               f"{result_path}/courts/court_mse")
                    write_json(have_court_dict, video_name,
                               f"{result_path}/courts/have_court")
                    write_json(players_dict, video_name,
                               f"{result_path}/players/player_kp")

                    pbar.update(1)

            # Release the video capture and writer objects
            video.release()

            try:
                # Code block that may raise exceptions
                # tracknet
                print("-" * 10 + "Starting Ball Detection" + "-" * 10)
                for res_root, res_dirs, res_files in os.walk(
                        f"{result_path}/videos/{video_name}"):
                    for res_file in res_files:
                        _, ext = os.path.splitext(res_file)
                        if ext.lower() in ['.mp4']:
                            res_video_path = os.path.join(res_root, res_file)
                            print(res_video_path)
                            ball_detect(res_video_path, f"{result_path}/ball")
                print("-" * 10 + "End Badminton Detection" + "-" * 10)
            except KeyboardInterrupt:
                print("Caught exception type on main.py ball_detect:",
                      type(KeyboardInterrupt).__name__)
                logging.basicConfig(filename='logs/error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
                logging.error(traceback.format_exc())
                exit()
            except Exception:
                print("Caught exception type on main.py ball_detect:",
                      type(Exception).__name__)
                logging.basicConfig(filename='logs/error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
                logging.error(traceback.format_exc())

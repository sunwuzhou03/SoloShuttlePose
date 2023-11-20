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
from utils import extract_numbers, write_json, read_json
from denoise import smooth
from event_detection import event_detect
import logging
import traceback

# from yolov5 detect.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_ball_position(img, original_img_=None):
    ret, thresh = cv2.threshold(img, 128, 1, 0)
    thresh = cv2.convertScaleAbs(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key=cv2.contourArea)

        if original_img_ is not None:
            # the contours are drawn here
            cv2.drawContours(original_img_, [c], -1, 255, 3)

        x, y, w, h = cv2.boundingRect(c)
        print("Center: ({}, {}) | Width: {} | Height: {}".format(x, y, w, h))

        return x, y, w, h


def ball_detect(video_path,result_path):
    imgsz = [288, 512]
    video_name = os.path.splitext(os.path.basename(video_path))[0]


    orivi_name, start_frame = extract_numbers(video_name)

    cd_save_dir= os.path.join(f"{result_path}/courts", f"court_kp")
    cd_json_path=f"{cd_save_dir}/{orivi_name}.json"
    court=read_json(cd_json_path)['court_info']            
            

    d_save_dir = os.path.join(result_path, f"loca_info/{orivi_name}")
    f_source = str(video_path)

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackNet().to(device)
    model.load_state_dict(torch.load("src/models/weights/ball_track.pt"))
    model.eval()

    vid_cap = cv2.VideoCapture(f_source)
    video_end = False
    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('{}/{}.mp4'.format(d_save_dir, video_name), fourcc,
    #                       fps, (w, h))

    count = 0
    with tqdm(total=video_len) as pbar:
        while vid_cap.isOpened():
            imgs = []
            for _ in range(3):
                ret, img = vid_cap.read()
                if not ret:
                    video_end = True
                    break

                imgs.append(img)

            if video_end:
                break

            imgs_torch = []
            for img in imgs:
                # https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_torch = torchvision.transforms.ToTensor()(img).to(
                    device)  # already [0, 1]
                img_torch = torchvision.transforms.functional.resize(
                    img_torch, imgsz, antialias=True)

                imgs_torch.append(img_torch)

            imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

            preds = model(imgs_torch)
            preds = preds[0].detach().cpu().numpy()

            y_preds = preds > 0.6
            y_preds = y_preds.astype('float32')
            y_preds = y_preds * 255
            y_preds = y_preds.astype('uint8')

            for i in range(3):
                if np.amax(y_preds[i]) <= 0:
                    ball_dict = {
                        f"{count + start_frame}": {
                            "visible": 0,
                            "x": 0,
                            "y": 0,
                        }
                    }
                    write_json(ball_dict, video_name, f"{d_save_dir}")

                    # cv2.imwrite('{}/{}.png'.format(d_save_dir, count), imgs[i])
                    # print('{} cx: 0  cy: 0'.format(count + start_frame))
                    # out.write(imgs[i])
                else:
                    pred_img = cv2.resize(y_preds[i], (w, h),
                                          interpolation=cv2.INTER_AREA)

                    # x, y, w, h = get_ball_position(pred_frame, original_img_=frames[i])
                    (cnts, _) = cv2.findContours(pred_img, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    rects = [cv2.boundingRect(ctr) for ctr in cnts]
                    max_area_idx = 0
                    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]

                    for ii in range(len(rects)):
                        area = rects[ii][2] * rects[ii][3]
                        if area > max_area:
                            max_area_idx = ii
                            max_area = area

                    target = rects[max_area_idx]
                    (cx_pred, cy_pred) = (int((target[0] + target[2] / 2)),
                                          int((target[1] + target[3] / 2)))

                    ball_dict = {
                        f"{count + start_frame}": {
                            "visible": 1,
                            "x": cx_pred,
                            "y": cy_pred,
                        }
                    }
                    write_json(ball_dict, video_name, f"{d_save_dir}")

                    # 绘图
                    # cv2.circle(imgs[i], (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                    # out.write(imgs[i])
                    # cv2.imwrite('{}/{}.png'.format(d_save_dir, count), imgs[i])
                    # print("{} cx: {}  cy: {}".format(count + start_frame,
                    #                                  cx_pred, cy_pred))

                count += 1
                pbar.update(1)

        while count < video_len:
            ball_dict = {
                f"{count + start_frame}": {
                    "visible": 0,
                    "x": 0,
                    "y": 0,
                }
            }
            write_json(ball_dict, video_name, f"{d_save_dir}")
            count += 1
            pbar.update(1)

    # denoise file save path
    dd_save_dir = os.path.join(result_path, f"loca_info(denoise)/{orivi_name}")
    os.makedirs(dd_save_dir, exist_ok=True)

    # json_path = f"{d_save_dir}/{video_name}.json"
    # smooth(json_path, dd_save_dir)
    
    # dd_json_path = f"{dd_save_dir}/{video_name}.json"
    # event_detect(dd_json_path, f"{result_path}")

    # smooth trajectory
    try:
        # Code block that may raise exceptions
        json_path = f"{d_save_dir}/{video_name}.json"
        smooth(json_path, court,dd_save_dir)
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
    
    # # capture hitting frame number
    # try:
    #     # Code block that may raise exceptions
    #     dd_json_path = f"{dd_save_dir}/{video_name}.json"
    #     event_detect(dd_json_path, f"{result_path}")
    #     # out.release()
    # except KeyboardInterrupt:
    #     print("Caught exception type on main.py ball_detect:",
    #             type(KeyboardInterrupt).__name__)
    #     logging.basicConfig(filename='logs/error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    #     logging.error(traceback.format_exc())
    #     exit()
    # except Exception:
    #     print("Caught exception type on main.py ball_detect:",
    #             type(Exception).__name__)
    #     logging.basicConfig(filename='logs/error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    #     logging.error(traceback.format_exc())
    

    vid_cap.release()

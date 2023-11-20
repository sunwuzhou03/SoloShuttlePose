import torch
import torchvision
import numpy as np
import copy
import cv2
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os
import sys

sys.path.append("src/tools")
sys.path.append("src/models")
from utils import read_json


class NetDetect(object):
    '''
    Tasks involving Keypoint RCNNs
    '''
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normal_net_info = None
        self.got_info = False
        self.mse = None
        self.setup_RCNN()

    def reset(self):
        self.got_info = False
        self.normal_net_info = None

    def setup_RCNN(self):
        self.__net_kpRCNN = torch.load('src/models/weights/net_kpRCNN.pth')
        self.__net_kpRCNN.to(self.device).eval()

    def del_RCNN(self):
        del self.__net_kpRCNN

    def pre_process(self, video_path, reference_path=None):
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)

        # Number of consecutively detected pitch frames
        last_count = 0
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        net_info_list = []
        # the number of skip frams per time
        skip_frames = max(int(fps) // 5, 5)

        # net detect don't need to do bisection search.
        if reference_path is not None:
            reference_data = read_json(reference_path)
            self.normal_net_info = reference_data['net_info']
            if self.normal_net_info is None:
                video.release()
                return total_frames
            self.__multi_points = self.__partition(
                self.normal_net_info).tolist()

            frame_number = reference_data.get('frame')
            if frame_number is None:
                frame_number = reference_data.get('first_rally_frame')
                if frame_number is None:
                    print('Error: frame number not found in reference data')
                    video.release()
                    sys.exit(1)

            print(
                f"video is pre-processing based on {reference_path} for net, frame number is {frame_number}"
            )
            video.release()
            return frame_number

        while True:
            # Read a frame from the video
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = video.read()

            if reference_path is not None:
                print(
                    f"video is pre-processing based on {reference_path} for net, current frame is {current_frame}"
                )
            else:
                print(
                    f"video is pre-processing for net, current frame is {current_frame}"
                )

            # If there are no more frames, break the loop
            if last_count >= skip_frames:
                if reference_path is None:
                    self.normal_net_info = net_info_list[skip_frames // 2]
                    for net_info in net_info_list:
                        if not self.__check_net(net_info):
                            self.normal_net_info = None
                            net_info_list = []
                            last_count = 0
                            print("Detect the wrong net!")
                            break

                if self.normal_net_info is not None:
                    return max(0, current_frame - 2 * skip_frames)
                else:
                    continue

            if not ret:
                # release the video
                video.release()
                return max(0, current_frame - 2 * skip_frames)

            net_info, have_net = self.get_net_info(frame)
            if have_net:
                last_count += 1
                net_info_list.append(net_info)
            else:
                if current_frame + skip_frames >= total_frames:
                    print(
                        "Fail to pre-process! Please to check the video or program!"
                    )
                    exit(0)
                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)
                last_count = 0
                net_info_list = []

    def __check_net(self, net_info):
        vec1 = np.array(self.normal_net_info)
        vec2 = np.array(net_info)
        mse = np.square(vec1 - vec2).mean()
        self.mse = mse
        if mse > 100:
            return False
        return True

    def get_net_info(self, img):
        self.__correct_points = None
        image = img.copy()
        self.mse = None
        frame_height, frame_weight, _ = image.shape
        image = F.to_tensor(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        output = self.__net_kpRCNN(image)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(
            output[0]['boxes'][high_scores_idxs],
            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        if len(output[0]['keypoints'][high_scores_idxs][post_nms_idxs]) == 0:
            self.got_info = False
            return None, self.got_info

        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][
                post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        self.__true_net_points = copy.deepcopy(keypoints[0])
        '''
        l -> left, r -> right, y = a * x + b
        '''

        self.__correct_points = self.__correction()

        # check if its value is normal
        if self.normal_net_info is not None:
            self.got_info = self.__check_net(self.__true_net_points)
            if not self.got_info:
                return None, self.got_info

        if self.normal_net_info is None:
            self.__multi_points = self.__partition(
                self.__correct_points).tolist()
        else:
            self.__multi_points = self.__partition(
                self.normal_net_info).tolist()

        self.got_info = True

        return self.__correct_points.tolist(), self.got_info

    def draw_net(self, image, mode="auto"):
        if self.normal_net_info is None and mode == "auto":
            # print("There is not net in the image! So you can't draw it.")
            return image
        elif mode == "frame_select":
            if self.__correct_points is None:
                return image
            self.__multi_points = self.__partition(
                self.__correct_points).tolist()

        image_copy = image.copy()
        c_edges = [[0, 1], [2, 3], [0, 4], [1, 5]]

        net_color_edge = (53, 195, 242)
        net_color_kps = (5, 135, 242)

        # draw the net
        for e in c_edges:
            cv2.line(image_copy, (int(self.__multi_points[e[0]][0]),
                                  int(self.__multi_points[e[0]][1])),
                     (int(self.__multi_points[e[1]][0]),
                      int(self.__multi_points[e[1]][1])),
                     net_color_edge,
                     2,
                     lineType=cv2.LINE_AA)

        for kps in [self.__multi_points]:
            for kp in kps:
                cv2.circle(image_copy, tuple(kp), 1, net_color_kps, 5)

        return image_copy

    def __correction(self):
        net_kp = np.array(self.__true_net_points)

        up_y = int((np.round(net_kp[0][1] + net_kp[3][1])) / 2)
        down_y = int((np.round(net_kp[1][1] + net_kp[2][1]) / 2))

        up_x = int(np.round((net_kp[0][0] + net_kp[1][0]) / 2))
        down_x = int(np.round((net_kp[3][0] + net_kp[2][0]) / 2))

        net_kp[0][1] = up_y
        net_kp[3][1] = up_y

        net_kp[1][1] = down_y
        net_kp[2][1] = down_y

        net_kp[0][0] = up_x
        net_kp[1][0] = up_x

        net_kp[3][0] = down_x
        net_kp[2][0] = down_x
        return net_kp

    def __partition(self, net_crkp):
        net_kp = np.array(net_crkp)

        p0 = net_kp[0]
        p1 = net_kp[3]

        p4 = net_kp[1]
        p5 = net_kp[2]

        p2 = np.array([p0[0], np.round((p4[1] + p0[1]) * (0.5))], dtype=int)
        p3 = np.array([p1[0], np.round((p5[1] + p1[1]) * (0.5))], dtype=int)

        kp = np.array([p0, p1, p2, p3, p4, p5], dtype=int)

        return kp

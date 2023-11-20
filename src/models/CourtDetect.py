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


class CourtDetect(object):
    '''
    Tasks involving Keypoint RCNNs
    '''
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normal_court_info = None
        self.got_info = False
        self.mse = None
        self.setup_RCNN()

    def reset(self):
        self.got_info = False
        self.normal_court_info = None

    def setup_RCNN(self):
        self.__court_kpRCNN = torch.load('src/models/weights/court_kpRCNN.pth')
        self.__court_kpRCNN.to(self.device).eval()

    def del_RCNN(self):
        del self.__court_kpRCNN

    def pre_process(self, video_path, reference_path=None):
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)

        # Number of consecutively detected pitch frames
        last_count = 0
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        court_info_list = []
        # the number of skip frams per time
        skip_frames = int(fps)

        if reference_path is not None:
            reference_data = read_json(reference_path)
            self.normal_court_info = reference_data['court_info']
            if self.normal_court_info is None:
                video.release()
                return total_frames

        while True:
            # Read a frame from the video
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = video.read()

            if reference_path is not None:
                print(
                    f"video is pre-processing based on {reference_path} for court, current frame is {current_frame}"
                )
            else:
                print(
                    f"video is pre-processing, current frame is {current_frame} for court"
                )

            # If there are no more frames, break the loop
            if last_count >= skip_frames:
                if reference_path is None:
                    self.normal_court_info = court_info_list[skip_frames // 2]
                    for court_info in court_info_list:
                        if not self.__check_court(court_info):
                            self.normal_court_info = None
                            court_info_list = []
                            last_count = 0
                            print("Detect the wrong court!")
                            break

                if self.normal_court_info is not None:
                    video.release()
                    return max(0, current_frame - 2 * skip_frames)
                elif not ret:
                    return total_frames
                else:
                    continue

            if not ret:
                video.release()
                return max(0, current_frame - 2 * skip_frames)

            court_info, have_court = self.get_court_info(frame)
            if have_court:
                last_count += 1
                court_info_list.append(court_info)
            else:
                if current_frame + skip_frames >= total_frames:
                    print(
                        "Fail to pre-process! Please to check the video or program!"
                    )
                    return total_frames

                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)
                last_count = 0
                court_info_list = []

    def __check_court(self, court_info):
        vec1 = np.array(self.normal_court_info)
        vec2 = np.array(court_info)
        mse = np.square(vec1 - vec2).mean()
        self.mse = mse
        if mse > 100:
            return False
        return True

    def get_court_info(self, img):
        image = img.copy()
        self.mse = None
        frame_height, frame_weight, _ = image.shape
        image = F.to_tensor(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.__court_kpRCNN(image)
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

        self.__true_court_points = copy.deepcopy(keypoints[0])
        '''
        l -> left, r -> right, y = a * x + b
        '''
        # correct to avoid the divide 0
        l_am = (self.__true_court_points[0][1] -
                self.__true_court_points[4][1])
        l_ad = (self.__true_court_points[0][0] -
                self.__true_court_points[4][0])
        l_a = l_am / (1 if l_ad == 0 else l_ad)
        l_b = self.__true_court_points[0][
            1] - l_a * self.__true_court_points[0][0]

        # correct to avoid the divide 0
        r_am = (self.__true_court_points[1][1] -
                self.__true_court_points[5][1])
        r_ad = (self.__true_court_points[1][0] -
                self.__true_court_points[5][0])
        r_a = r_am / (1 if r_ad == 0 else r_ad)

        r_b = self.__true_court_points[1][
            1] - r_a * self.__true_court_points[1][0]
        mp_y = (self.__true_court_points[2][1] +
                self.__true_court_points[3][1]) / 2

        self.__court_info = [l_a, l_b, r_a, r_b, mp_y]

        self.__correct_points = self.__correction()

        # check if current court information get from the normal camera view
        if self.normal_court_info is not None:
            self.got_info = self.__check_court(self.__correct_points)
            if not self.got_info:
                return None, self.got_info

        if self.normal_court_info is None:
            self.__multi_points = self.__partition(
                self.__correct_points).tolist()
        else:
            self.__multi_points = self.__partition(
                self.normal_court_info).tolist()

        keypoints[0][0][0] -= 80
        keypoints[0][0][1] -= 80
        keypoints[0][1][0] += 80
        keypoints[0][1][1] -= 80
        keypoints[0][2][0] -= 80
        keypoints[0][3][0] += 80
        keypoints[0][4][0] -= 80
        keypoints[0][4][1] = min(keypoints[0][4][1] + 80, frame_height - 40)
        keypoints[0][5][0] += 80
        keypoints[0][5][1] = min(keypoints[0][5][1] + 80, frame_height - 40)

        self.__extended_court_points = keypoints[0]

        self.got_info = True

        # return self.__true_court_points, self.got_info
        return self.__correct_points.tolist(), self.got_info

    def draw_court(self, image, mode="auto"):
        if not self.got_info and mode == "auto":
            print("There is not court in the image! So you can't draw it.")
            return image
        elif mode == "frame_select":
            if self.__correct_points is None:
                return image
            self.__multi_points = self.__partition(
                self.__correct_points).tolist()

        image_copy = image.copy()
        c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4],
                   [3, 8], [4, 9], [5, 6], [5, 10], [6, 7], [6, 11], [7, 8],
                   [7, 12], [8, 9], [8, 13], [9, 14], [10, 11], [10, 15],
                   [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],
                   [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18],
                   [17, 22], [18, 19], [18, 23], [19, 24], [20, 21], [20, 25],
                   [21, 22], [21, 26], [22, 23], [22, 27], [23, 24], [23, 28],
                   [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],
                   [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32],
                   [32, 33], [33, 34]]
        court_color_edge = (53, 195, 242)
        court_color_kps = (5, 135, 242)

        # draw the court
        for e in c_edges:
            cv2.line(image_copy, (int(self.__multi_points[e[0]][0]),
                                  int(self.__multi_points[e[0]][1])),
                     (int(self.__multi_points[e[1]][0]),
                      int(self.__multi_points[e[1]][1])),
                     court_color_edge,
                     2,
                     lineType=cv2.LINE_AA)
        for kps in [self.__multi_points]:
            for kp in kps:
                cv2.circle(image_copy, tuple(kp), 1, court_color_kps, 5)

        return image_copy

    def __correction(self):
        court_kp = np.array(self.__true_court_points)
        ty = np.round((court_kp[0][1] + court_kp[1][1]) / 2)
        my = (court_kp[2][1] + court_kp[3][1]) / 2
        by = np.round((court_kp[4][1] + court_kp[5][1]) / 2)
        court_kp[0][1] = ty
        court_kp[1][1] = ty
        court_kp[2][1] = my
        court_kp[3][1] = my
        court_kp[4][1] = by
        court_kp[5][1] = by
        return court_kp

    def __partition(self, court_crkp):
        court_kp = np.array(court_crkp)
        tlspace = np.array([
            np.round((court_kp[0][0] - court_kp[2][0]) / 3),
            np.round((court_kp[2][1] - court_kp[0][1]) / 3)
        ],
                           dtype=int)
        trspace = np.array([
            np.round((court_kp[3][0] - court_kp[1][0]) / 3),
            np.round((court_kp[3][1] - court_kp[1][1]) / 3)
        ],
                           dtype=int)
        blspace = np.array([
            np.round((court_kp[2][0] - court_kp[4][0]) / 3),
            np.round((court_kp[4][1] - court_kp[2][1]) / 3)
        ],
                           dtype=int)
        brspace = np.array([
            np.round((court_kp[5][0] - court_kp[3][0]) / 3),
            np.round((court_kp[5][1] - court_kp[3][1]) / 3)
        ],
                           dtype=int)

        p2 = np.array(
            [court_kp[0][0] - tlspace[0], court_kp[0][1] + tlspace[1]])
        p3 = np.array(
            [court_kp[1][0] + trspace[0], court_kp[1][1] + trspace[1]])
        p4 = np.array([p2[0] - tlspace[0], p2[1] + tlspace[1]])
        p5 = np.array([p3[0] + trspace[0], p3[1] + trspace[1]])

        p8 = np.array(
            [court_kp[2][0] - blspace[0], court_kp[2][1] + blspace[1]])
        p9 = np.array(
            [court_kp[3][0] + brspace[0], court_kp[3][1] + brspace[1]])
        p10 = np.array([p8[0] - blspace[0], p8[1] + blspace[1]])
        p11 = np.array([p9[0] + brspace[0], p9[1] + brspace[1]])

        kp = np.array([
            court_kp[0], court_kp[1], p2, p3, p4, p5, court_kp[2], court_kp[3],
            p8, p9, p10, p11, court_kp[4], court_kp[5]
        ],
                      dtype=int)

        ukp = []

        for i in range(0, 13, 2):
            sub2 = np.round((kp[i] + kp[i + 1]) / 2)
            sub1 = np.round((kp[i] + sub2) / 2)
            sub3 = np.round((kp[i + 1] + sub2) / 2)
            ukp.append(kp[i])
            ukp.append(sub1)
            ukp.append(sub2)
            ukp.append(sub3)
            ukp.append(kp[i + 1])
        ukp = np.array(ukp, dtype=int)
        return ukp

    def player_detection(self, outputs):
        boxes = outputs[0]['boxes'].cpu().detach().numpy()
        filtered_joint = []
        joints = outputs[0]['keypoints'].cpu().detach().numpy()
        in_court_indices = self.__check_in_court_instances(joints)

        if in_court_indices:
            conform, combination = self.__check_top_bot_court(
                in_court_indices, boxes)
            if conform:
                filtered_joint.append(
                    joints[in_court_indices[combination[0]]].tolist())
                filtered_joint.append(
                    joints[in_court_indices[combination[1]]].tolist())
                filtered_joint = self.__top_bottom(filtered_joint)

                for points in filtered_joint:
                    for i, joints in enumerate(points):
                        points[i] = joints[0:2]

                return (True, filtered_joint)
            else:
                return (False, None)
        else:
            return (False, None)

    def __top_bottom(self, joint):
        a = joint[0][-1][1] + joint[0][-2][1]
        b = joint[1][-1][1] + joint[1][-2][1]
        # the first player on the top of court, swap
        if a >= b:
            joint[0], joint[1] = joint[1], joint[0]
        return joint

    def __check_top_bot_court(self, indices, boxes):
        '''
        check if up court and bot court got player
        
        if detect the player left the court, the get_court_info will return False  even if it detects the court.  

        To some degree, it will impact on getting player's posture data.
        '''
        court_mp = self.__court_info[4]
        for i in range(len(indices)):
            combination = 1
            if boxes[indices[0]][1] < court_mp < boxes[
                    indices[combination]][3]:
                return True, [0, combination]
            elif boxes[indices[0]][3] > court_mp > boxes[
                    indices[combination]][1]:
                return True, [0, combination]
            else:
                combination += 1
        return False, [0, 0]

    def __check_in_court_instances(self, joints):
        indices = []
        for i in range(len(joints)):
            if self.__in_court(joints[i]):
                indices.append(i)
        return None if len(indices) < 2 else indices

    def __in_court(self, joint):
        '''
        check if player is in court
        '''
        l_a = self.__court_info[0]
        l_b = self.__court_info[1]
        r_a = self.__court_info[2]
        r_b = self.__court_info[3]

        ankle_x = (joint[15][0] + joint[16][0]) / 2
        ankle_y = (joint[15][1] + joint[16][1]) / 2

        top = ankle_y > self.__extended_court_points[0][1]
        bottom = ankle_y < self.__extended_court_points[5][1]

        lmp_x = (ankle_y - l_b) / l_a
        rmp_x = (ankle_y - r_b) / r_a
        left = ankle_x > lmp_x
        right = ankle_x < rmp_x

        if left and right and top and bottom:
            return True
        else:
            return False

import cv2
import torch
import torchvision
import numpy as np
import copy
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F


class PoseDetect:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_RCNN()

    def reset(self):
        self.got_info = False

    def setup_RCNN(self):
        self.__pose_kpRCNN = torch.load('src/models/weights/pose_kpRCNN.pth')
        self.__pose_kpRCNN.to(self.device).eval()

    def del_RCNN(self):
        del self.__pose_kpRCNN

    def get_human_joints(self, frame):
        frame_copy = frame.copy()
        outputs = self.__human_detection(frame_copy)
        human_joints = outputs[0]['keypoints'].cpu().detach().numpy()
        filtered_outputs = []
        for i in range(len(human_joints)):
            filtered_outputs.append(human_joints[i].tolist())

        for points in filtered_outputs:
            for i, joints in enumerate(points):
                points[i] = joints[0:2]
        filtered_outputs
        return outputs, filtered_outputs

    def __human_detection(self, frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        t_image = transforms.Compose(
            [transforms.ToTensor()])(pil_image).unsqueeze(0).to(self.device)
        outputs = self.__pose_kpRCNN(t_image)
        return outputs

    def draw_key_points(self, filtered_outputs, image, human_limit=-1):
        image_copy = image.copy()
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
                 (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
                 (12, 14), (14, 16), (5, 6)]
        
        # top player is blue and bottom one is red
        top_color_edge = (255, 0, 0)
        bot_color_edge = (0, 0, 255)
        top_color_joint = (115, 47, 14)
        bot_color_joint = (35, 47, 204)


        for i in range(len(filtered_outputs)):

            if i > human_limit and human_limit != -1:
                break

            color = top_color_edge if i == 0 else bot_color_edge
            color_joint = top_color_joint if i == 0 else bot_color_joint

            keypoints = np.array(filtered_outputs[i])  # 17, 2
            keypoints = keypoints[:, :].reshape(-1, 2)
            for p in range(keypoints.shape[0]):
                cv2.circle(image_copy,
                           (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3,
                           color_joint,
                           thickness=-1,
                           lineType=cv2.FILLED)

            for e in edges:
                cv2.line(image_copy,
                         (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         color,
                         2,
                         lineType=cv2.LINE_AA)
        return image_copy

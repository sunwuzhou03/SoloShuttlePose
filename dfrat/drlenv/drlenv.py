import cv2
import numpy as np
import pandas as pd


'''
Why choose the deep reinforcement learning for this task?

Just For Fun.

action: 
0: skip forward fps
1: skip forward one frame
2: skip backward fps
3: skip backward one frame 
4: prediction

state:
5 frames
if the fifth frame is labeled the valid frame, and the [2,3,4,5,6], [3,4,5,6,7], [4,5,6,7,8] will be label the valid frame.
'''

class BadmintonVideoEnv:
    def __init__(self,video_path,annotation_path) -> None:
        self.video=cv2.VideoCapture(video_path)
        self.annotation=pd.csv(annotation_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def reset(self):
        pass

    def step(self,action):
        '''
        return
        (frames, frame number vector)-> next_state
        reward -> sum(the real frame number - current frame number) / total_number 
        '''
        pass

    def render(self):
        pass



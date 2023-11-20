import cv2
import os


class VideoClip(object):
    def __init__(self,
                 video_name,
                 fps,
                 total_frames,
                 frame_width,
                 frame_height,
                 save_path="./") -> None:
        self.video_name = video_name
        self.save_path = save_path
        self.fps = fps
        self.skip_frames = int(fps // 2)
        self.total_frames = total_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_list = []
        self.begin = 0
        self.end = 1
        self.no_court_cnt = 0

    def __setup(self):
        self.frame_list = []

    def add_frame(self, have_court, frame, frame_count):
        if frame_count == self.total_frames - 1:
            if len(self.frame_list) < int(self.fps * 0.5):
                self.frame_list.clear()
                self.begin = -1
                self.end = 0
                return False
            self.end = frame_count
            self.frame_list.append(frame)
            self.__make_video()
            self.__setup()
            self.no_court_cnt = 0
            return True

        if have_court:
            if self.begin == -1:
                self.begin = frame_count
            self.frame_list.append(frame)
            return False
        elif not have_court:
            # when the valid frame in frame_list less than 0.5 seconds run in ori-fps
            if len(self.frame_list) < int(self.fps * 0.5):
                self.frame_list.clear()
                self.begin = -1
                self.end = 0
                return False
            # when the valid frame in frame_list more than 3 seconds run in ori-fps
            elif len(self.frame_list) > int(self.fps * 3):
                # to check if it's break by accident
                if self.no_court_cnt >= self.skip_frames:
                    self.end = frame_count
                    self.__make_video()
                    self.__setup()
                    self.no_court_cnt = 0
                    return True
                else:
                    self.frame_list.append(frame)
                    self.no_court_cnt += 1
                    return False

    def __make_video(self):

        # 设置输出视频的名称、编解码器、帧率和视频分辨率
        video_name = f"{self.video_name}_{self.begin}-{self.end-1}.mp4"
        full_path = os.path.join(self.save_path, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.fps
        output_video_format = (self.frame_width, self.frame_height)

        video_writer = cv2.VideoWriter(full_path, fourcc, fps,
                                       output_video_format)

        # 遍历列表中的每个元素，将元素绘制到图像上，并将图像写入到视频中
        for frame in self.frame_list:
            video_writer.write(frame)

        # 释放资源
        video_writer.release()
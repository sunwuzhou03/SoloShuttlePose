import math
import cv2
from PIL import Image, ImageTk
import os
import copy
import tkinter as tk
from tkinter import messagebox
import argparse
import sys

sys.path.append("src/tools")
sys.path.append("src/models")
from utils import write_json, clear_file, is_video_detect, find_next, find_reference
from VideoClip import VideoClip
from PoseDetect import PoseDetect
from CourtDetect import CourtDetect
from NetDetect import NetDetect


def on_closing():
    # Handle window close events
    # Here you can close the window or perform other actions
    global user_choice
    user_choice = False
    small_window.destroy()


def on_keypress(event):
    # Handle key events
    global user_choice
    if event.keysym == 'Escape':
        # If the ESC key was pressed, close the window or perform another action
        user_choice = False
        small_window.destroy()


def yes_button_click():
    global user_choice
    user_choice = True
    os.remove(reference_path)
    small_window.destroy()


def no_button_click():
    global user_choice
    user_choice = False
    small_window.destroy()


# Image display callback function
def update_image():
    # Read video frames
    global frame, frame_counter, video_name, court_info, net_info
    global new_height, new_width
    print(f"for {video_name}, current frame is {frame_counter}")
    ret, frame = video.read()

    if not ret:
        # Video readout complete.
        return
    frame = frame.astype('uint8')
    h, w = new_height, new_width

    # # Resize an image to a specified size
    # frame_resized = cv2.resize(frame.copy(), (w, h))
    # # Converting OpenCV images to PIL images
    # frame_pil = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # frame_pil = Image.fromarray(frame_pil)
    # # Show unprocessed images in the left-hand tab
    # frame_tk1 = ImageTk.PhotoImage(frame_pil)
    # image_label1.configure(image=frame_tk1)
    # image_label1.image = frame_tk1

    # Display the processed image in the right-hand tab
    # use CourtDetect and PoseDetect to process frame

    frame_copy = frame.copy()
    court_info, have_court = court_detect.get_court_info(frame_copy)
    net_info, have_net = net_detect.get_net_info(frame_copy)
    # if have_court and have_net:
    #     net_info[1][1],net_info[2][1]=court_info[2][1],court_info[3][1]

    if have_court:
        original_outputs, human_joints = pose_detect.get_human_joints(frame)
        have_player, players_joints = court_detect.player_detection(
            original_outputs)

        if have_player:
            court_frame = court_detect.draw_court(frame)
            net_frame = net_detect.draw_net(court_frame, "frame_select")
            frame_copy = pose_detect.draw_key_points(players_joints, net_frame)

    frame_processed = cv2.resize(frame_copy, (w, h))
    frame_processed_pil = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
    frame_processed_pil = Image.fromarray(frame_processed_pil)
    frame_tk2 = ImageTk.PhotoImage(frame_processed_pil)

    image_label.configure(image=frame_tk2)
    image_label.image = frame_tk2

    # Update Window Display
    window.update()


# Keyboard event callback functions
def key_press(event):
    global frame_counter
    global total_frames
    global frame
    global video_name
    global video_path
    global court_info
    global net_info

    key = event.keysym

    if key == "Return":  # Press the Enter key to save the image of the current frame
        refer_dict = {
            "frame": frame_counter,
            "court_info": court_info,
            "net_info": net_info
        }
        write_json(refer_dict, video_name, "references")

        print(f"save the reference information for {video_path}")
        window.destroy()
        return

    if key == "Escape":  # Press Esc to exit the program
        window.destroy()
        return

    if key == "Up":  # Press up. Jump 30 frames forward.
        frame_counter -= 30
        frame_counter = max(0, frame_counter)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

    if key == "Down":  # Press down. Jump 30 frames back.
        frame_counter += 30
        frame_counter = min(frame_counter, total_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

    if key == "Left":  # Press left. Jump forward one frame.
        frame_counter -= 1
        frame_counter = max(0, frame_counter)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

    if key == "Right":  # Press right. Jump back one frame.
        frame_counter += 1
        frame_counter = min(frame_counter, total_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

    # update image
    update_image()


parser = argparse.ArgumentParser(description='para transfer')
parser.add_argument('--folder_path',
                    type=str,
                    default="videos",
                    help='folder_path -> str type.')
args = parser.parse_args()
print(args)

folder_path = args.folder_path

# for select
video_name = None
user_choice = True

# for write json
court_info = None
net_info = None

new_width = 1280
new_height = 720

for root, dirs, files in os.walk(folder_path):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in ['.mp4']:

            video_path = os.path.join(root, file)
            video_name = os.path.basename(video_path).split('.')[0]

            user_choice = True
            reference_path = find_reference(video_name)
            while reference_path is not None and user_choice:
                # Create a window
                small_window = tk.Tk()
                small_window.title("delete file")
                small_window.state('zoomed')

                small_window.protocol("WM_DELETE_WINDOW",
                                      on_closing)  # 将关闭事件连接到处理函数上

                # 将按键事件绑定到处理函数上
                small_window.bind('<Key>', on_keypress)

                # Display the name of the file
                label = tk.Label(small_window,
                                 text="file name: " + reference_path)
                label.grid(row=0, column=0, columnspan=2, pady=10)

                # Create a "Yes" button
                yes_button = tk.Button(small_window,
                                       text="yes",
                                       command=yes_button_click)
                yes_button.grid(row=1, column=0, padx=10)

                # Create a "No" button
                no_button = tk.Button(small_window,
                                      text="no",
                                      command=no_button_click)
                no_button.grid(row=1, column=1, padx=10)

                small_window.mainloop()
                reference_path = find_reference(video_name)

            if not user_choice:
                continue

            # Open the video file
            video = cv2.VideoCapture(video_path)

            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # Save current frame counter
            frame_counter = total_frames // 2
            frame = None
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

            court_detect = CourtDetect()
            pose_detect = PoseDetect()
            net_detect = NetDetect()

            # Create a window
            window = tk.Tk()

            # 获取屏幕分辨率
            screen_width = int(window.winfo_screenwidth()*0.85)
            screen_height = int(window.winfo_screenheight()*0.85)

            # 计算适合屏幕的大小
            frame_ratio = frame_width / frame_height
            screen_ratio = screen_width / screen_height

            if frame_ratio >= screen_ratio:
                # 如果帧的宽高比大于或等于屏幕宽高比，则根据屏幕宽度缩放帧
                new_width = screen_width
                new_height = math.floor(new_width / frame_ratio)
            else:
                # 否则根据屏幕高度缩放帧
                new_height = screen_height
                new_width = math.floor(new_height * frame_ratio)

            # screen_width = window.winfo_screenwidth()
            # screen_height = window.winfo_screenheight()
            # width_ratio = 1
            # height_ratio = 1
            # new_width = int(frame_width * width_ratio)
            # new_height = int(frame_height * height_ratio)

            window.title(f"Select valid frame from {video_name}")
            # window.geometry("800x600+200+100")
            window.state('zoomed')

            # Create a right side image label
            image_label = tk.Label(window)
            image_label.pack()

            update_image()
            

            # Binding Keyboard Event Handler Functions
            window.bind("<Key>", key_press)

            # Enter the main loop.
            window.mainloop()

            # Close the video file
            video.release()

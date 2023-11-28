import os
import shutil
import yt_dlp
import pandas as pd
import numpy as np
import os
import sys
import cv2

sys.path.append("src/tools")
from utils import read_json, write_json, extract_numbers

target_directory = "ShuttleSet\ShuttleSet22\data4drl"

# for get url
# dir_name="Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals"
# sub_dir=os.path.join(target_directory,dir_name)
# video_path=os.path.join(sub_dir,f"{dir_name}.mp4")
# try:
#     if not os.path.exists(video_path):
#         search_name = dir_name.replace("_", " ")
#         command = f'yt-dlp --write-link --min-sleep-interval 10 --max-sleep-interval 30 "ytsearch:{search_name}" -f 137 --restrict-filenames -o "{video_path}"'
        
#         os.system(command)
# except:
#     pass
# exit()
   
# limit_num为30时表示遍历的前30个不下载，只下载以后的
limit_num=0#30
video_cnt=0

frame_len=5

# 遍历目录下的子目录
for dir in os.listdir(target_directory):
    sub_dir=os.path.join(target_directory, dir)
    if os.path.isdir(sub_dir):
        dir_name = os.path.basename(dir)
        if video_cnt==0:
            video_cnt+=1
            continue
        print(dir_name)
        if os.path.exists(os.path.join(sub_dir,"annotations.json")):
            os.remove(os.path.join(sub_dir,"annotations.json"))
            print("Delete the {}".format(os.path.join(sub_dir,"annotations.json")))
        
        csv_path_list=[]
        for file_name in os.listdir(sub_dir):
            # 检查文件扩展名是否为csv
            if file_name.endswith('.csv'):
                # file_path = os.path.join(folder_path, file_name)
                csv_path=os.path.join(sub_dir,file_name)
                csv_path_list.append(csv_path)
        
        import random
        random.seed(3407)
        if len(csv_path_list)>5:
            random_paths = random.sample(csv_path_list, 5)
        else:
            continue

        for csv_path in random_paths:
            df=pd.read_csv(csv_path)
            frame_dict={}
            for index,row in df.iterrows():
                if str(row['pos']) == 'nan':
                    print(str(row['pos']))
                    continue
                frame_dict[str(row['frame_num'])]=row['pos']
                write_json(frame_dict,"annotations",sub_dir)
                frame_dict={}
        video_path=os.path.join(sub_dir,f"{dir_name}.mp4")
        # url_path=os.path.join(sub_dir,f"{dir_name}.url")

        if not os.path.exists(video_path):
            search_name = dir_name.replace("_", " ")
            command = f'yt-dlp --write-link --min-sleep-interval 10 --max-sleep-interval 30 "ytsearch:{search_name}" -f 137 --restrict-filenames -o "{video_path}"'
            
            os.system(command)


        json_path=os.path.join(sub_dir,f"annotations.json")

        label_dict=read_json(json_path)
        video=cv2.VideoCapture(video_path)
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total_frames)
        if total_frames<=0:
            continue

        top_path=os.path.join(sub_dir,f"{dir_name}/top")
        bottom_path=os.path.join(sub_dir,f"{dir_name}/bottom")
        none_path=os.path.join(sub_dir,f"{dir_name}/none")

        os.makedirs(top_path,exist_ok=True)
        os.makedirs(bottom_path,exist_ok=True)
        os.makedirs(none_path,exist_ok=True)
        
        frame_buffer=[]
        frame_number_buffer=[]

        negtive=0
        positive=0
        from tqdm import tqdm
        with tqdm(total=total_frames) as pbar:
            while True:

                label="none"
                current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                ret,frame=video.read()
                
                    
                if not ret:
                    break
                
                

                frame_buffer.append(frame)
                frame_number_buffer.append(current_frame)


                if len(frame_number_buffer)>=frame_len:

                    if str(frame_number_buffer[1]) in label_dict.keys():
                        label=label_dict[str(str(frame_number_buffer[1]))]
                    
                    if str(frame_number_buffer[2]) in label_dict.keys():
                        label=label_dict[str(frame_number_buffer[2])]

                    if str(frame_number_buffer[3]) in label_dict.keys():
                        label=label_dict[str(frame_number_buffer[3])]


                    if label == "none":
                        if negtive-positive<20:
                            save_path=os.path.join(none_path,str(frame_number_buffer[0]))
                            os.makedirs(save_path,exist_ok=True)
                            for i in range(frame_len):
                                cv2.imwrite(f'{save_path}/{frame_number_buffer[i]}.png', frame_buffer[i])
                            negtive+=1
                    elif label=="top":
                        save_path=os.path.join(top_path,str(frame_number_buffer[0]))
                        os.makedirs(save_path,exist_ok=True)
                        for i in range(frame_len):
                            cv2.imwrite(f'{save_path}/{frame_number_buffer[i]}.png', frame_buffer[i])
                        positive+=1
                    else:
                        save_path=os.path.join(bottom_path,str(frame_number_buffer[0]))
                        os.makedirs(save_path,exist_ok=True)
                        for i in range(frame_len):
                            cv2.imwrite(f'{save_path}/{frame_number_buffer[i]}.png', frame_buffer[i])
                        positive+=1
                    frame_buffer.pop(0)
                    frame_number_buffer.pop(0)
                pbar.update(1)
        video.release()

        os.remove(video_path)
        
        print("-----------------")
import os
import glob
import shutil
import sys
import pandas as pd
import numpy as np
import warnings
import shutil
warnings.simplefilter('ignore')

sys.path.append("src/tools")
from utils import read_json, write_json

import math

def edist(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

loca4match="E:/SoloShuttlePoseRes/res/ball/loca_info(denoise)"
player4match="E:/SoloShuttlePoseRes/res/players/player_kp"
action4match="ShuttleSet/ShuttleSet22/match_db"
data4drl="ShuttleSet/ShuttleSet22/data4drl"
court4match="E:/SoloShuttlePoseRes/res/courts/court_kp"
video4drl="res/videos"
url4drl="ShuttleSet/ShuttleSet22/match_db"

result_path="res"
make_data4drl=False

if make_data4drl:

    for dir in os.listdir(data4drl):
        if os.path.isdir(os.path.join(data4drl, dir)):
            dir_name = os.path.basename(dir)
            data_path=os.path.join(data4drl,dir_name)
            for csv_path in glob.glob(os.path.join(data_path, "*.csv")):
                os.remove(csv_path)
            for json_path in glob.glob(os.path.join(data_path, "*.json")):
                os.remove(json_path)
    # exit()
    def dist_to_pose(pose, p):
        pose = pose.reshape(17, 2)
        p = p.reshape(1, 2)
        D = np.sum((pose - p) * (pose - p), axis=1)
        return min(D)


    for dir in os.listdir(action4match):


        if os.path.isdir(os.path.join(action4match, dir)):
            dir_name = os.path.basename(dir)

            print(dir_name)


            
            # video_json_path=os.path.join(video4drl,dir_name)
            # for json_path in glob.glob(os.path.join(video_json_path, "*.json")):
            #     shutil.copy2(json_path, final_df_path)

            # url_path=os.path.join(url4drl,dir_name)
            # for url_path in glob.glob(os.path.join(url_path, "*.url")):
            #     shutil.copy2(url_path, final_df_path)

            
            loca_path=os.path.join(loca4match,dir_name)
            if not os.path.exists(loca_path):
                continue
            
            final_df_path=os.path.join(data4drl,dir_name)
            os.makedirs(final_df_path,exist_ok=True)

            loca_dict={}
            for json_path in glob.glob(os.path.join(loca_path, "*.json")):
                loca_dict.update(read_json(json_path))
            
            court_dict=read_json(os.path.join(court4match,f"{dir_name}.json"))
            
            player_dict=read_json(os.path.join(player4match,f"{dir_name}.json"))
            action_path=os.path.join(action4match,dir_name)


            for csv_path in glob.glob(os.path.join(action_path, "*.csv")):
                
                csv_name = os.path.basename(csv_path).split('.')[0]

                df = pd.read_csv(csv_path)
                
                

                # 创建一个新的DataFrame存储修改后的数据
                new_df = []
                hit_list=[]
                nhit = 0
                hits = np.array([0]*len(df))
                # Majority vote for who started the rally
                # If hit number is odd, then whoever started the rally
                # is the opposite of whoever was detected.
                votes = [0, 0]
                for index, row in df.iterrows():
                    frame_num = row['frame_num']
                    
                    court_info=court_dict['court_info']
                    top_l=abs(court_info[0][1]-court_info[2][1])
                    bottom_l=abs(court_info[2][1]-court_info[4][1])

                    if str(frame_num) not in loca_dict.keys():
                        # print("There is not info about ball location. ")
                        continue
                        
                    
                    row['ball'] = [loca_dict[str(frame_num)]['x'],loca_dict[str(frame_num)]['y']]
                    
                    # top player 姿态插值
                    bf=frame_num-12
                    ef=frame_num+12
                    while str(bf) not in loca_dict.keys():
                        bf+=1
                    while str(ef) not in loca_dict.keys():
                        ef-=1
                    ef=ef+1

                    wrong=False
                    if player_dict[str(frame_num)]['top'] is None or player_dict[str(frame_num)]['bottom'] is None:
                        top_kps=[[] for _ in range(bf,ef)]

                        for k in range(17):
                            x_ls=[]
                            y_ls=[]
                            for fn in range(bf,ef):
                                if str(fn) in player_dict.keys():
                                    if player_dict[str(fn)]['top'] is not None:
                                        x_ls.append(player_dict[str(fn)]['top'][k][0])
                                        y_ls.append(player_dict[str(fn)]['top'][k][1])
                                    else:
                                        x_ls.append(float('nan'))
                                        y_ls.append(float('nan'))      
                            try:             
                                x_pd=pd.DataFrame(x_ls.copy()).interpolate(method='slinear')
                                y_pd=pd.DataFrame(y_ls.copy()).interpolate(method='slinear')
                            except:
                                x_pd=pd.DataFrame(x_ls.copy())
                                y_pd=pd.DataFrame(y_ls.copy())
                                wrong=True
                                break
                                

                            x_pd.ffill(inplace=True)
                            x_pd.bfill(inplace=True)
                            
                            y_pd.ffill(inplace=True)
                            y_pd.bfill(inplace=True)
                        
                            for fn in range(bf,ef):
                                # c=player_dict[str(fn)]['top'][k]
                                # print(x_pd)
                                top_kps[fn-bf].append([x_pd.loc[fn-bf,0],y_pd.loc[fn-bf,0]])
                                            
                        for fn in range(bf,ef):
                            player_dict[str(fn)]['top']=top_kps[fn-bf].copy()
                        

                        bottom_kps=[[] for _ in range(bf,ef)]
                        for k in range(17):
                            x_ls=[]
                            y_ls=[]
                            for fn in range(bf,ef):
                                if str(fn) in player_dict.keys():
                                    if player_dict[str(fn)]['bottom'] is not None:
                                        x_ls.append(player_dict[str(fn)]['bottom'][k][0])
                                        y_ls.append(player_dict[str(fn)]['bottom'][k][1])
                                    else:
                                        x_ls.append(float('nan'))
                                        y_ls.append(float('nan'))      
                                            
                            try:             
                                x_pd=pd.DataFrame(x_ls.copy()).interpolate(method='slinear')
                                y_pd=pd.DataFrame(y_ls.copy()).interpolate(method='slinear')
                            except:
                                x_pd=pd.DataFrame(x_ls.copy())
                                y_pd=pd.DataFrame(y_ls.copy())
                                wrong=True
                                break

                            x_pd.ffill(inplace=True)
                            x_pd.bfill(inplace=True)
                            
                            y_pd.ffill(inplace=True)
                            y_pd.bfill(inplace=True)
                            for fn in range(bf,ef):
                                bottom_kps[fn-bf].append([x_pd.loc[fn-bf,0],y_pd.loc[fn-bf,0]])
                            
                        for fn in range(bf,ef):
                            player_dict[str(fn)]['bottom']=bottom_kps[fn-bf].copy()

                    if frame_num>ef or frame_num<bf:
                        print("bf and bf is wrong") 
                        print(frame_num,bf,ef)                   
                    try:
                        ball=(loca_dict[str(frame_num)]['x'],loca_dict[str(frame_num)]['y'])

                        row['top'] = player_dict[str(frame_num)]['top'].copy()
                        row['bottom'] = player_dict[str(frame_num)]['bottom'].copy()


                        top_kp15,top_kp16=row['top'][15].copy(),row['top'][16].copy()
                        top_kp17=[(top_kp15[0]+top_kp16[0])/2,(top_kp15[1]+top_kp16[1])/2]
                        top_kp=row['top'].copy()
                        row['top'].append(top_kp17)

                        bottom_kp15,bottom_kp16=row['bottom'][15].copy(),row['bottom'][16].copy()
                        bottom_kp17=[(bottom_kp15[0]+bottom_kp16[0])/2,(bottom_kp15[1]+bottom_kp16[1])/2]
                        bottom_kp=row['bottom'].copy()
                        row['bottom'].append(bottom_kp17)


                        db = dist_to_pose(np.array(bottom_kp), np.array([row['ball'][0],row['ball'][1]]))
                        dt = dist_to_pose(np.array(top_kp), np.array([row['ball'][0],row['ball'][1]]))
                            
                        if db < dt:
                            person = 1
                        else:
                            person = 2
                        
                        if nhit % 2:
                            person = 3 - person
                        votes[person - 1] += 1
                        nhit += 1
                        # 将修改后的row添加到新的DataFrame中
                        new_df.append(row)    
                    except:
                        wrong=True
                        print(csv_path)
                    if wrong:
                        break
                if wrong:
                    continue
                last = 2 if votes[0] > votes[1] else 1
                # if (match, rally) in manual_label:
                #     last = 3 - manual_label[match, rally]
                    
                for i in range(hits.shape[0]):
                    hits[i] = 3 - last
                    last = hits[i]
                            
                new_df = pd.DataFrame(new_df)
                for index, row in new_df.iterrows():
                    if hits[index] == 1:
                        new_df.loc[index, 'pos'] = 'bottom'
                    else:
                        new_df.loc[index, 'pos'] = 'top'

                columns_to_copy = ["rally", "ball_round", "time", "frame_num", "player", "pos","type", "lose_reason", "getpoint_player", "ball", "top", "bottom"]

                final_df = pd.DataFrame(new_df, columns=columns_to_copy)

                if len(final_df)==0:
                    print(csv_path)
                else:
                    final_df.to_csv(f"{final_df_path}/{csv_name}.csv", index=False, encoding="utf-8")
                    
        
import re
def extract_numbers(filename):
    pattern = r"(\w+)_(\d+)-\d+"
    match = re.match(pattern, filename)
    if match:
        name = str(match.group(1))
        start = int(match.group(2))
        return name, start
    else:
        return filename, 0

def get_pattern(csv_path):
    df=pd.read_csv(csv_path)
    
    for i in range(len(df)):
        pos=df.loc[i,'pos']
        player=df.loc[i,'player']
        if pos == 'top':
            return player
    if len(df)==1:
        pos=df.loc[0,'pos']
        player=df.loc[0,'player']
        if pos == 'top':
            return player
        else:
            return 3-player
        

for dir in os.listdir(data4drl):
    if os.path.isdir(os.path.join(data4drl, dir)):
        dir_name = os.path.basename(dir)
        data_path=os.path.join(data4drl,dir_name)
        # 将DataFrame保存为Excel文件
        # df = pd.DataFrame([], columns=['round','player','pos'])
        # filename = 'apos.xlsx'  # 文件名

        # # 保存DataFrame到Excel文件
        # df.to_excel(os.path.join(data_path,filename), index=False)

        # print(data_path)
        ls11=[]
        ls12=[]
        py1t=0

        ls21=[]
        ls22=[]
        py2t=0

        ls31=[]
        ls32=[]
        py3t=0

        for csv_path in glob.glob(os.path.join(data_path, "*.csv")):
            csv_name=os.path.basename(csv_path).split('.')[0]
            _,rally_number=extract_numbers(csv_name)
            if dir_name!="Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals":
                # exit()
                pass
            # print(csv_path)
            if rally_number==1:
                top_player=get_pattern(csv_path)
                # print(top_player)
                if py1t==0:
                    py1t=top_player
                
                if len(ls11)==0:
                    ls11.append(csv_path)
                elif top_player==py1t:
                    ls11.append(csv_path)
                else:
                    ls12.append(csv_path)

            elif rally_number==2:
                top_player=get_pattern(csv_path)
                if py2t==0:
                    py2t=top_player
                if len(ls21)==0:
                    ls21.append(csv_path)
                elif top_player==py2t:
                    ls21.append(csv_path)
                else:
                    ls22.append(csv_path)

            elif rally_number==3:
                top_player=get_pattern(csv_path)
                if py3t==0:
                    py3t=top_player
                if len(ls31)==0:
                    ls31.append(csv_path)
                elif top_player==py3t:
                    ls31.append(csv_path)
                else:
                    ls32.append(csv_path)

        print(1,dir_name,len(ls11),len(ls12))
        if len(ls11)<len(ls12):
            print(ls11)
            pass
        else:
            print(ls12)
            pass
        
        print(2,dir_name,len(ls21),len(ls22))
        if len(ls21)<len(ls22):
            print(ls21)
            pass
        else:
            print(ls22)
            pass

        print(3,dir_name,len(ls31),len(ls32))
        if len(ls31)<len(ls32):
            print(ls31)
            pass
        else:
            print(ls32)
            pass
    

for dir in os.listdir(data4drl):
    if os.path.isdir(os.path.join(data4drl, dir)):
        dir_name = os.path.basename(dir)
        data_path=os.path.join(data4drl,dir_name)

        mutual_pos_path=os.path.join(data_path,"apos.xlsx")
        pos_df=pd.read_excel(mutual_pos_path)
        # print(pos_df)

        round_pos=[{str(pos_df.loc[0,'player']):str(pos_df.loc[0,'pos'])},\
                   {str(pos_df.loc[1,'player']):str(pos_df.loc[1,'pos'])},\
                    {str(pos_df.loc[2,'player']):str(pos_df.loc[2,'pos'])}]
        
        for csv_path in glob.glob(os.path.join(data_path, "*.csv")):
            csv_name=os.path.basename(csv_path).split('.')[0]
            _,round_number=extract_numbers(csv_name)
            df=pd.read_csv(csv_path)

            one_pos='top'
            two_pos='bottom'
            # print(round_pos[round_number-1])
            for key,value in round_pos[round_number-1].items():
                if one_pos!=str(value):
                    one_pos,two_pos=two_pos,one_pos
            for index,row in df.iterrows():
                if str(row['player']) in round_pos[round_number-1].keys(): 
                    df.loc[index,'pos']=one_pos
                else:
                    df.loc[index,'pos']=two_pos
            
            df.to_csv(csv_path, index=False, encoding="utf-8")
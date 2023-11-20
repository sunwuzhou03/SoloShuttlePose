import os
import glob
import shutil
import sys
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter('ignore')

sys.path.append("src/tools")
from utils import read_json, write_json

import math

def edist(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

loca4match="E:/SoloShuttlePoseRes/res/ball/loca_info(denoise)"
player4match="E:/SoloShuttlePoseRes/res/players/player_kp"
court4match="E:/SoloShuttlePoseRes/res/courts/court_kp"
hit4match="ShuttleSet/ShuttleSet22/data4drl"
result_path="ShuttleSet/ShuttleSet22/data4acreg"

shutil.rmtree(result_path)
os.makedirs(result_path,exist_ok=True)

def dist_to_pose(pose, p):
    pose = pose.reshape(17, 2)
    p = p.reshape(1, 2)
    D = np.sum((pose - p) * (pose - p), axis=1)
    return min(D)


for dir in os.listdir(hit4match):
    if os.path.isdir(os.path.join(hit4match, dir)):
        dir_name = os.path.basename(dir)
        print(dir_name)
        loca_path=os.path.join(loca4match,dir_name)
        if not os.path.exists(loca_path):
            continue
        
        final_df_path=os.path.join(result_path,dir_name)
        os.makedirs(final_df_path,exist_ok=True)

        loca_dict={}
        for json_path in glob.glob(os.path.join(loca_path, "*.json")):
            loca_dict.update(read_json(json_path))
        
        court_dict=read_json(os.path.join(court4match,f"{dir_name}.json"))
        
        player_dict=read_json(os.path.join(player4match,f"{dir_name}.json"))
        action_path=os.path.join(hit4match,dir_name)


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
            hit_list=df['frame_num'].values.tolist()
            type_list=df['type'].values.tolist()
            pos_list=df['pos'].values.tolist()
            
            try:
                begin_frame=hit_list[0]
                end_frame=hit_list[-1]
            except:
                print(csv_path)    
                # exit()

            
            bf=begin_frame-12
            ef=end_frame+12
            while str(bf) not in loca_dict.keys():
                bf+=1
            while str(ef) not in loca_dict.keys():
                ef-=1
            ef=ef+1

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
                                
                x_pd=pd.DataFrame(x_ls.copy()).interpolate(method='slinear')
                y_pd=pd.DataFrame(y_ls.copy()).interpolate(method='slinear')
                
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
                                
                x_pd=pd.DataFrame(x_ls.copy()).interpolate(method='slinear')
                y_pd=pd.DataFrame(y_ls.copy()).interpolate(method='slinear')
                
                x_pd.ffill(inplace=True)
                x_pd.bfill(inplace=True)
                
                y_pd.ffill(inplace=True)
                y_pd.bfill(inplace=True)
            

            
                for fn in range(bf,ef):

                    bottom_kps[fn-bf].append([x_pd.loc[fn-bf,0],y_pd.loc[fn-bf,0]])
                
            for fn in range(bf,ef):
                player_dict[str(fn)]['bottom']=bottom_kps[fn-bf].copy()
                    




            final_df=pd.DataFrame(columns=['frame', 'top', 'bottom','court','net','ball','pos','type'])                                    
            for fn in range(bf,ef):
                if str(fn) not in loca_dict.keys():
                    break
                row=[]
                row.append(fn)
                row.append(player_dict[str(fn)]['top'])
                row.append(player_dict[str(fn)]['bottom'])
                row.append(court_dict['court_info'])
                row.append(court_dict['net_info'])
                row.append([loca_dict[str(fn)]['x'],loca_dict[str(fn)]['y']])
                if fn in hit_list:
                    row.append(pos_list[hit_list.index(fn)])
                    row.append(type_list[hit_list.index(fn)])
                    
                else:
                    row.append('None')
                    row.append('None')
                final_df.loc[len(final_df)] = row

            x=[loca_info[0] for loca_info in final_df['ball']]
            y=[loca_info[1] for loca_info in final_df['ball']]

            
            pre_dif = []
            save_flag=True
            for i in range(0, len(x)):
                if i == 0:
                    pre_dif.append(0)
                else:
                    pre_dif.append(
                        ((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)**(1 / 2))
                    if pre_dif[-1]>200:
                        save_flag=False
                        break
            
            if not save_flag:
                continue

            final_df.to_csv(f"{final_df_path}/{csv_name}.csv", index=False, encoding="utf-8")
            


import pandas as pd
import numpy as np
import copy
import logging
import traceback
class Trajectory(object):
    def __init__(self, df, interp=True):
        # Get poses and trajectories
        trajectory = df.copy() #pd.read_csv(filename)
        try:
            if interp:
                trajectory[trajectory.X == 0] = float('nan')
                trajectory[trajectory.Y == 0] = float('nan')
                trajectory = trajectory.assign(X_pred=trajectory.X.interpolate(method='slinear'))
                trajectory = trajectory.assign(Y_pred=trajectory.Y.interpolate(method='slinear'))

                trajectory.ffill(inplace=True)
                trajectory.bfill(inplace=True)

                # to avoid all loca are (0,0)
                trajectory.fillna(0, inplace=True)

                Xb, Yb = trajectory.X_pred.tolist(), trajectory.Y_pred.tolist()
            else:
                Xb, Yb = trajectory.X.tolist(), trajectory.Y.tolist()
        except Exception as e:
            # import sys
            # print(sys.exc_info()[0])  # 输出错误类型
            # print(sys.exc_info()[1])  # 输出错误信息
            # 配置日志输出
            trajectory.fillna(value=0, inplace=True)
            Xb, Yb = trajectory.X.tolist(), trajectory.Y.tolist()
            # 配置日志输出
            logging.basicConfig(filename='logs/error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
            # 记录完整异常信息
            logging.error(traceback.format_exc())
            
            

        self.X = Xb
        self.Y = Yb

def read_trajectory_3d(file_path):
    return pd.read_csv(str(file_path))

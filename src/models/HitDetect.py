import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving import hdf5_format
import h5py
from pathlib import Path
import sys
sys.path.append("src/tools")


class Pose:
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (6, 12), (5, 11), (11, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle" ]

    def __init__(self, kplines=[], fullPose=False):
        if not kplines:
            return

        keypoints = []
        self.score = 0
        for kp, line in enumerate(kplines):
            if not fullPose:
                i, px, py, score = [float(x) for x in line.split()]
            else:
                px, py, score = [float(x) for x in line.split()]
                i = kp
            keypoints.append((int(i), np.array([px, py])))
            self.score += score
        self.init_from_kp(keypoints)

    def init_from_kparray(self, kparray):
        kp = np.array(kparray).reshape((17, 2))
        keypoints = []
        for i in range(17):
            keypoints.append((i, kp[i]))
        self.init_from_kp(keypoints)

    # Each pose has 17 key points, representing the skeleton
    def init_from_kp(self, keypoints):
        # Keypoints should be tuples of (id, point)
        self.kp = np.empty((17, 2))
        self.kp[:] = np.NaN

        for i, p in keypoints:
            self.kp[i] = p

        self.bx = [min(self.kp[:, 0]), max(self.kp[:, 0])]
        self.by = [min(self.kp[:, 1]), max(self.kp[:, 1])]


    def draw_skeleton(self, img, colour=(0, 128, 0), thickness=5):
        cimg = img.copy()
        for line in self.skeleton:
            X, Y = self.kp[line[0]], self.kp[line[1]]
            if any(np.isnan(X)) or any(np.isnan(Y)):
                continue
            # We sometimes fill in NaNs with zeros
            if sum(X) == 0 or sum(Y) == 0:
                continue
            p0, p1 = tuple(X.astype(int)), tuple(Y.astype(int))
            # For the legs, colour them and the ankles separately
            if line == (13, 15) or line == (14, 16):
                cimg = cv2.line(cimg, p0, p1, (0, 128, 128), thickness)
                cimg = cv2.circle(cimg, p1, 3, (128, 128, 0), thickness=-1)
            else:
                cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg

    def get_base(self):
        # Returns the midpoint of the two ankle positions
        # Returning one of the two points if theres a NaN
        # or a zero
        left_nan = self.kp[15][0] != self.kp[15][0] or self.kp[15][0] == 0
        right_nan = self.kp[16][0] != self.kp[16][0] or self.kp[16][0] == 0
        if left_nan:
            return self.kp[16]
        elif right_nan:
            return self.kp[15]
        elif left_nan and right_nan:
            return self.get_centroid()
        return (self.kp[15] + self.kp[16]) / 2.

    def get_centroid(self):
        n = 0
        p = np.zeros((2,))
        for i in range(17):
            if any(np.isnan(self.kp[i])) or max(self.kp[i]) == 0:
                continue

            n += 1
            p += self.kp[i]
        return p / n

    def can_reach(self, p, epsx=1.5, epsy=1.5):
        # if within (1+/-eps) of the bounding box then we can reach it
        dx, dy = self.bx[1] - self.bx[0], self.by[1] - self.by[0]
        return self.bx[0] - epsx * dx < p[0] < self.bx[1] + epsx * dx and \
               self.by[0] - epsy * dy < p[1] < self.by[1] + epsy * dy

class HitDetector(object):
    def __init__(self, court, poses, trajectory):
        self.court = court
        self.poses = poses
        self.trajectory = trajectory

    # Returns the hits in the given trajectory as well as who hit it
    # Output is are two lists of values:
    #   - List of frame ids where things are hit
    #   - 0 (no hit), 1 (bottom player hits), 2 (top player hits)
    def detect_hits(self):
        pass

class AdhocHitDetector(HitDetector):
    def __init__(self, poses, trajectory):
        super().__init__(None, poses, trajectory)

    def _detect_hits_1d(self, z, thresh=4, window=8):
        # For a hit to be registered, the point must be a local max / min and
        # the slope must exceed thresh on either the left or right side
        # The slope is averaged by the window parameter to remove noise
        z = np.array(z)
        bpts = []
        for i in range(window+1, len(z)-window-1):
            if (z[i]-z[i-1]) * (z[i]-z[i+1]) < 0:
                continue

            # This is a local opt
            left = abs(np.median(z[i-window+1:i+1] - z[i-window:i]))
            right = abs(np.median(z[i+1:i+window+1] - z[i:i+window]))
            if max(left, right) > thresh:
                bpts.append(i)
        return bpts

    def _merge_hits(self, x, y, closeness=2):
        bpt = []
        for t in sorted(x + y):
            if len(bpt) == 0 or bpt[-1] < t - closeness:
                bpt.append(t)
        return bpt

    def _detect_hits(self, x, y, thresh=10, window=7, closeness=15):
        return self._merge_hits(
            self._detect_hits_1d(x, thresh, window),
            self._detect_hits_1d(y, thresh, window),
            closeness
        )

    def detect_hits(self, fps=25):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        result = self._detect_hits(Xb, Yb)

        # Filter hits by pose
        is_hit = []
        last_hit = -1
        # Filter hits by velocity
        avg_hit = np.average(np.diff(result))
        for i, fid in enumerate(result):
            if i+1 < len(result) and result[i+1] - fid > 1.6 * avg_hit:
                is_hit.append(0)
                continue

            if fid > self.poses[0].values.shape[0]:
                break

            c = np.array([Xb[fid], Yb[fid]])
            reached_by = 0
            dist_reached = 1e99
            for j in range(2):
                xy = self.poses[j].iloc[fid].to_list()
                pose = Pose()
                pose.init_from_kparray(xy)
                if pose.can_reach(c):
                    pdist = np.linalg.norm(c - pose.get_centroid())
                    if not reached_by or reached_by == last_hit or pdist < dist_reached:
                        reached_by = j + 1
                        dist_reached = pdist

            if reached_by:
                last_hit = reached_by
            is_hit.append(reached_by)

        print('Total shots hit by players:', sum(x > 0 for x in is_hit))
        print('Total impacts detected:', len(result))
        print('Distribution of shot times:')
        plt.hist(np.diff(result))
        print('Average time between shots (s):', np.average(np.diff(result)) / fps)
        return result, is_hit
    

# def scale_data(x):
#     x = np.array(x)
#     def scale_by_col(x, cols, eps=1e-6):
#         x_ = np.array(x[:, cols])
#         idx = np.abs(x_) < eps
#         m, M = np.min(x_[~idx]), np.max(x_[~idx])
#         x_[~idx] = (x_[~idx] - m) / (M - m) + 1
#         x[:, cols] = x_
#         return x

#     even_cols = [2*i for i in range(x.shape[1] // 2)]
#     odd_cols = [2*i+1 for i in range(x.shape[1] // 2)]
#     x = scale_by_col(x, even_cols)
#     x = scale_by_col(x, odd_cols)
#     return x

# class MLHitDetector(HitDetector):
#     @staticmethod
#     def create_model(feature_dim, num_consecutive_frames):
#         input_layer = Input(shape=(feature_dim,))
#         X = input_layer
#         X = Reshape(
#             target_shape=(num_consecutive_frames, feature_dim // num_consecutive_frames))(X)
#         # Two layers of bidirectional grus
#         X = Bidirectional(GRU(64, return_sequences=True))(X)
#         X = Bidirectional(GRU(64, return_sequences=True))(X)
#         X = GlobalMaxPool1D()(X)
#         X = Dense(3)(X)
#         X = Softmax()(X)
#         output_layer = X
#         model = Model(input_layer, output_layer)
#         return model

#     def __init__(self, court, poses, trajectory, model_path, fps=25, debug=True):
#         super().__init__(court, poses, trajectory)

#         self.fps = fps
#         self.debug = debug
#         with h5py.File(model_path, mode='r') as f:
#             self.temperature = f.attrs['temperature']
#             #self.model = hdf5_format.load_model_from_hdf5(f)
#             #self.model = MLHitDetector.create_model(2418, 31) # 31-13-13
#             self.model = MLHitDetector.create_model(936, 12) # 12-6-0
#             self.model.load_weights(model_path)

#         import tensorflow.keras.backend as K

#         trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
#         non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])

#         if debug:
#             print('Number of layers:', len(self.model.layers))
#             print('Total params: %d' % (trainable_count + non_trainable_count))
#             print('Trainable params: %d' % trainable_count)
#             print('Non-trainable params: %d' % non_trainable_count)

#     def naive_postprocessing(self, y_pred, detect_thresh=0.1):
#         Xb, Yb = self.trajectory.X, self.trajectory.Y
#         court_pts = self.court.corners
#         num_consec = int(self.model.input_shape[1] // (2 * (34 + 4 + 1)))

#         detections = np.where(y_pred[:,0] < detect_thresh)[0]
#         result, clusters, who_hit = [], [], []
#         min_x, max_x = np.min(court_pts, axis=0)[0], np.max(court_pts, axis=0)[0]
#         for t in detections:
#             # Filter based on time
#             if len(clusters) == 0 or clusters[-1][0] < t - self.fps / 2:
#                 clusters.append([t])
#             else:
#                 clusters[-1].append(t)

#         delta = 0.1 * (max_x - min_x)
#         for cluster in clusters:
#             # Filter based on whether any part of the cluster is outside
#             any_out = False
#             votes = np.array([0.] * y_pred.shape[1])
#             for t in cluster:
#                 if Xb[t] < min_x + delta or Xb[t] > max_x - delta:
#                     any_out = True
#                     break
#                 votes += y_pred[t]
#             if not any_out:
#                 # Detections start around 6 frames from the end
#                 gap = 4
#                 result.append(int(np.median(cluster) + num_consec - gap))
#                 who_hit.append(int(np.argmax(votes)))

#         is_hit = []
#         avg_hit = np.average(np.diff(result))
#         last_hit, last_time = -1, -1
#         to_delete = [0] * len(result)
#         for i, fid in enumerate(result):
#             if i >= len(who_hit):
#                 break

#             # Another filter: prevent two hits in a row by the same person within 0.8s
#             if fid - last_time < 0.8 * self.fps and last_hit == who_hit[i]:
#                 to_delete[i] = 1
#                 continue

#             is_hit.append(who_hit[i])
#             last_time = fid
#             last_hit = who_hit[i]

#         result = [r for i, r in enumerate(result) if not to_delete[i]]
#         return result, is_hit

#     def dp_postprocessing(self, y_pred):
#         tau = .9 * np.mean(y_pred[y_pred[:, 0] < 0.1, 1:3])
#         score = y_pred - tau
#         # Smooth out scores a bit so that we're more likely to hit the centre of a hit window
#         for i in range(3):
#             score[:, i] = np.convolve(score[:, i], np.ones(2)/2, mode='same')

#         N = y_pred.shape[0]
#         T = int(1.2 * N // self.fps)
#         D = int(self.fps // 2)

#         # N x T dp table.
#         # dp[i, j, k] := best score we get on frames [0, i) with j hits left, with k hitting last
#         dp = (-1e99) * np.ones((N + D, T + 1, 2))
#         dp[0:D, :, :] = 0

#         choice = np.zeros((N + D, T + 1, 2))
#         for i in range(D, N + D):
#             for j in range(T):
#                 for k in range(2):
#                     # Can choose not to hit on this frame
#                     cval = dp[i-1, j, k]
#                     if dp[i, j, k] <= cval:
#                         dp[i, j, k] = cval
#                         choice[i, j, k] = 0

#                     if j >= 0:
#                         # Can choose to hit on current frame
#                         # TODO: This transition is technically not correct
#                         # for the first 15 frames or so (because of boundary conditions)
#                         # but the edge case doesnt come up in the test data set so we'll
#                         # leave a TODO and fix it later.
#                         cval = dp[i - D, j + 1, (k ^ 1)] + score[i - D, 1 + k]
#                         if dp[i, j, k] <= cval:
#                             dp[i, j, k] = cval
#                             choice[i, j, k] = 1

#         # Now we reconstruct the hits and the hit times
#         result, is_hit = [], []
#         best = -1e99

#         i, j, k = 0, 0, 0
#         for sj in range(T + 1):
#             for sk in range(2):
#                 if dp[N + D - 1, sj, sk] > best:
#                     i, j, k = N + D - 1, sj, sk
#                     best = dp[N + D - 1, sj, sk]

#         while i >= D:
#             if choice[i, j, k] == 0:
#                 # Nothing happened, move back one frame
#                 i -= 1
#             else:
#                 gap = 8
#                 # Hit case
#                 result.append(i + gap - D)
#                 is_hit.append(k + 1)
#                 i = i - D
#                 j += 1
#                 k ^= 1
#         return list(reversed(result)), list(reversed(is_hit))

#     def detect_hits(self):
#         Xb, Yb = self.trajectory.X, self.trajectory.Y
#         num_consec = int(self.model.input_shape[1] // (2 * (34 + 4 + 1)))
#         court_pts = self.court.corners
#         corner_coords = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

#         bottom_player = self.poses[0]
#         top_player = self.poses[1]

#         corners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

#         x_list = []
#         L = min(bottom_player.values.shape[0], len(Xb))
#         for i in range(num_consec):
#             end = L-num_consec+i+1
#             x_bird = np.array(list(zip(Xb[i:end], Yb[i:end])))
#             x_pose = np.hstack([bottom_player.values[i:end], top_player.values[i:end]])
#             x = np.hstack([x_bird, x_pose, np.array([corners for j in range(i, end)])])

#             x_list.append(x)
#         x_inp = np.hstack(x_list)
#         x_inp = scale_data(x_inp)

#         compute_logits = K.function([self.model.layers[0].input], [self.model.layers[-2].output])
#         y_pred = tf.nn.softmax(compute_logits(x_inp)[0] / self.temperature).numpy()
#         # Use this line if there is no temperature
#         # y_pred = self.model.predict(x_inp)

#         if self.debug:
#             print('Sum of predicted scores:', np.sum(y_pred, axis=0))

#         # result, is_hit = self.naive_postprocessing(y_pred)
#         result, is_hit = self.dp_postprocessing(y_pred)

#         if self.debug:
#             num_hits = sum(x > 0 for x in is_hit)
#             print('Total shots hit by players:', num_hits)
#             if num_hits:
#                 print('Percentage of shots hit by player 1:', sum(x == 1 for x in is_hit) / num_hits)
#             else:
#                 print('No hits detected.')
#             print('Total impacts detected:', len(result))
#             print('Distribution of shot times:')
#             plt.hist(np.diff(result))
#             print('Average time between shots (s):', np.average(np.diff(result)) / self.fps)
#         return result, is_hit

def event_detect(df):

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.ndimage
    from scipy.optimize import curve_fit
    import csv
    from mpl_toolkits.mplot3d import Axes3D
    import math
    from scipy.signal import find_peaks
    import argparse
    import os
    import sys


    def angle(v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle


    def get_point_line_distance(point, line):
        point_x = point[0]
        point_y = point[1]
        line_s_x = line[0]
        line_s_y = line[1]
        line_e_x = line[2]
        line_e_y = line[3]
        if line_e_x - line_s_x == 0:
            return math.fabs(point_x - line_s_x)
        if line_e_y - line_s_y == 0:
            return math.fabs(point_y - line_s_y)
        #斜率
        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        #截距
        b = line_s_y - k * line_s_x
        #带入公式得到距离dis
        dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
        return dis


    # df=pd.DataFrame()
    list1 = []
    points = []
    frames = []
    realx = []
    realy = []
    count=0
    num=0
    start_frame=0
    for index,row in df.iterrows():
        list1.append([index,1,row['ball'][0],row['ball'][1]])
    front_zeros = np.zeros(len(list1))
    for i in range(len(list1)):
        frames.append(int(float(list1[i][0])))
        realx.append(int(float(list1[i][2])))
        realy.append(int(float(list1[i][3])))
        if int(float(list1[i][2])) != 0:
            front_zeros[num] = count
            points.append((int(float(list1[i][2])), int(float(list1[i][3])),
                           int(float(list1[i][0]))))
            num += 1
        else:
            count += 1

    # some video don't have badminton location information
    if num == 0:
        print("There is not any hitting event in this video!")
        print()
        return

    # 羽球2D軌跡點
    points = np.array(points)
    x, y, z = points.T

    Predict_hit_points = np.zeros(len(frames))
    ang = np.zeros(len(frames))
    # from scipy.signal import find_peaks
    peaks, properties = find_peaks(y, prominence=5)

    # print curve peaks
    # print(peaks)

    if (len(peaks) >= 5):
        lower = np.argmin(y[peaks[0]:peaks[1]])
        if (y[peaks[0]] - lower) < 5:
            peaks = np.delete(peaks, 0)

        lower = np.argmin(y[peaks[-2]:peaks[-1]])
        if (y[peaks[-1]] - lower) < 5:
            peaks = np.delete(peaks, -1)

    print()
    print('Begin : ', end='')
    start_point = 0

    for i in range(len(y) - 1):
        if ((y[i] - y[i + 1]) / (z[i + 1] - z[i]) >= 5):
            start_point = i + front_zeros[i]
            Predict_hit_points[int(start_point)] = 1
            print(int(start_point) + start_frame)
            break

    end_point = 10000

    print('Predict points : ')
    plt.plot(z, y * -1, '-')
    for i in range(len(peaks)):
        print(peaks[i] + int(front_zeros[peaks[i]] + start_frame), end=',')
        if (peaks[i] + int(front_zeros[peaks[i]]) >= start_point
                and peaks[i] + int(front_zeros[peaks[i]]) <= end_point):
            Predict_hit_points[peaks[i] + int(front_zeros[peaks[i]])] = 1

    #打擊的特定frame = peaks[i]+int(front_zeros[peaks[i]])
    print()
    print('Extra points : ')
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1] + 1
        upper = []
        plt.plot(z[start:end], y[start:end] * -1, '-')
        lower = np.argmin(y[start:end])  #找到最低谷(也就是從最高點開始下墜到下一個擊球點),以此判斷扣殺或平球軌跡
        for j in range(start + lower, end + 1):
            if (j - (start + lower) > 5) and (end - j > 5):
                if (y[j] - y[j - 1]) * 3 < (y[j + 1] - y[j]):
                    print(j + start_frame, end=',')
                    ang[j + int(front_zeros[j])] = 1

                point = [x[j], y[j]]
                line = [x[j - 1], y[j - 1], x[j + 1], y[j + 1]]
                # if get_point_line_distance(point,line) > 2.5:
                if angle([x[j - 1], y[j - 1], x[j], y[j]],
                         [x[j], y[j], x[j + 1], y[j + 1]]) > 130:
                    print(j + start_frame, end=',')
                    ang[j + int(front_zeros[j])] = 1

    ang, _ = find_peaks(ang, distance=15)
    #final_predict, _  = find_peaks(Predict_hit_points, distance=10)
    for i in ang:
        Predict_hit_points[i] = 1
    Predict_hit_points, _ = find_peaks(Predict_hit_points, distance=5)
    final_predict = []
    for i in (Predict_hit_points):
        final_predict.append(i)

    print()
    print('Final predict : ')
    for pred in list(final_predict):
        print(pred + start_frame, end=",")

    print()
    if len(list(final_predict))>0:
        print(f'End : {list(final_predict)[-1]+ start_frame}')
    else:
        print("End : ")
    
    return final_predict

import pandas as pd
import os
import random

special_path="ShuttleSet\ShuttleSet22\data4acreg\He_Bing_Jiao_Chen_Yu_Fei_GWANGJU_YONEX_Korea_Masters_2022_Final/rally_3-18.csv"

def random_csv_file(folder_path):
    csv_files = []
    
    # 遍历文件夹及其子文件夹，找到所有的CSV文件路径
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    
    # 随机选择一个CSV文件路径
    if len(csv_files) > 0:
        random_path = random.choice(csv_files)
        return random_path
    else:
        return None

# 指定文件夹路径
folder_path = "ShuttleSet\ShuttleSet22\data4acreg"

# data_path="ShuttleSet\ShuttleSet22\data4acreg\Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals/rally_1-1.csv"

# 调用函数进行随机抽取
data_path = random_csv_file(folder_path)

# 定义处理函数
def parse_list(x):
    return eval(x)

# 读取CSV文件并转换为DataFrame，同时将 'my_list' 列中的列表解析
data = pd.read_csv(data_path, converters={"ball": parse_list,"top":parse_list,"bottom":parse_list,"court":parse_list,"net":parse_list})

X=[loca_info[0] for loca_info in data['ball']]
Y=[loca_info[1] for loca_info in data['ball']]

poses=[data[['top']],data[['bottom']]]

trajectory=pd.DataFrame(columns=['X','Y'])
trajectory.X=X
trajectory.Y=Y

court=data.loc[0,'court']
net=data.loc[0,'net']

begin_frame=data.loc[0,'frame']

detect1=AdhocHitDetector(poses,trajectory)

hit_true=[]
for index,row in data.iterrows():
    if str(row['type']) != "nan":
        hit_true.append(index)

result, is_hit=detect1.detect_hits()
print(data_path)
y_pred0=event_detect(data)

print(f"y_pred0={y_pred0}")
print(f"y_pred1={result}")
print(f"y_true={hit_true}")

# for data in result:
#     print(data+begin_frame)
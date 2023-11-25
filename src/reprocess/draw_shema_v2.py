    
import pandas as pd
import matplotlib.pyplot as plt
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

# 调用函数进行随机抽取
data_path = random_csv_file(folder_path)

# data_path="ShuttleSet\ShuttleSet22\dataset/valid\AN_Se_Young_Gregoria_Mariska_TUNJUNG_Malaysia_Masters_2022_SemiFinals/rally_1-1.csv"


if data_path:
    print("The random csv path:", data_path)
else:
    print("There is not csv file in this folder.")
    exit()


# 定义处理函数
def parse_list(x):
    return eval(x)


# 读取CSV文件并转换为DataFrame，同时将 'my_list' 列中的列表解析
data = pd.read_csv(data_path, converters={"ball": eval,"top":eval,"bottom":eval,"court":eval,"net":eval})

X=[loca_info[0] for loca_info in data['ball']]
Y=[loca_info[1] for loca_info in data['ball']]

poses=[data[['top']],data[['bottom']]]

trajectory=pd.DataFrame(columns=['X','Y'])
trajectory.X=X
trajectory.Y=Y

court=data.loc[0,'court']
net=data.loc[0,'net']

image_list=[]
fkey=-1
for index,row in data.iterrows():
    # figure 2
    plt.figure(figsize=(16, 12)) 
    plt.ylim(0, 1080)
    plt.xlim(0, 1920)
    plt.gca().invert_yaxis()

    # 给定的点
    players_joints=row['top']

    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in players_joints]
    y = [joint[1] for joint in players_joints]

    # 创建散点图
    plt.scatter(x, y,c="b")

    edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)]

    # 循环添加标号
    for i, joint in enumerate(players_joints):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'b-')

    players_joints=row['bottom']

    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in players_joints]
    y = [joint[1] for joint in players_joints]

    # 创建散点图
    plt.scatter(x, y,c="r")

    edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)]

    # 循环添加标号
    for i, joint in enumerate(players_joints):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'r-')


    # 场
    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in court]
    y = [joint[1] for joint in court]
    # 创建散点图
    plt.scatter(x, y,c="y")
    edges = [(0, 1), (2, 3), (4, 5),(0,4),(1,5)]
    # 循环添加标号
    for i, joint in enumerate(court):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')


    # 网
    # 提取 x 坐标和 y 坐标
    x = [joint[0] for joint in net]
    y = [joint[1] for joint in net]
    # 创建散点图
    plt.scatter(x, y,c="y")
    edges = [(0, 1), (1, 2), (2, 3),(0,3)]
    # 循环添加标号
    for i, joint in enumerate(net):
        plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # 绘制连接线
    for edge in edges:
        plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')

    # 球

    ball=row['ball']
    plt.scatter(ball[0], ball[1],c="r")
    plt.annotate("ball", (ball[0], ball[1]), textcoords="offset points", xytext=(0,10), ha='center')



    # 设置图形标题和轴标签
    plt.title('All_info')
    plt.xlabel('X')
    plt.ylabel('Y')

    from PIL import Image
    import copy

    # 使用PIL库加载图像文件，并将其添加到图像列表中
    plt.savefig('Frame.png')
    image_list.append(Image.open('Frame.png').copy())

    plt.clf()
    plt.close()
import os
os.remove("Frame.png")
# 保存为GIF文件
image_list[0].save('docs/imgs/random.gif', save_all=True, append_images=image_list[1:], duration=66, loop=0)

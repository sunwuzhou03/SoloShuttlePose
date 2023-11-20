import os

def count_csv_files(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                count += 1
    return count

folder_path = 'ShuttleSet/ShuttleSet22/data4drl'  # 替换为你的文件夹路径
csv_count = count_csv_files(folder_path)
print(f"There are {csv_count} CSV files in the folder {folder_path} and its subfolders.")
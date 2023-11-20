import os
import shutil
import yt_dlp


source_directory = "ShuttleSet/ShuttleSet22/match"
target_directory = "ShuttleSet/ShuttleSet22/match_db"

# 确保要删除的文件夹存在
if os.path.exists(target_directory):
    pass
else:
   shutil.copytree(source_directory, target_directory)
   print(f"create {target_directory}")
   
# limit_num为30时表示遍历的前30个不下载，只下载以后的
limit_num=0#30
video_cnt=0

# 遍历目录下的子目录
for dir in os.listdir(target_directory):
    if os.path.isdir(os.path.join(target_directory, dir)):
        dir_name = os.path.basename(dir)
        search_name = dir_name.replace(".", "_")
        search_name = search_name.replace("-", "_")
        os.rename(os.path.join(target_directory, dir_name),
                  os.path.join(target_directory, search_name))

# download video
for dir in os.listdir(target_directory):
    target_path=os.path.join(target_directory, dir)
    # print(os.path.isdir(os.path.join(target_directory, dir)))
    if os.path.isdir(target_path):
        dir_name = os.path.basename(dir)
        print(dir_name)
        video_cnt+=1
        if video_cnt<=limit_num:
            continue
        search_name = dir.replace("_", " ")

        # 构建命令行
        command = f'yt-dlp --write-link --min-sleep-interval 10 --max-sleep-interval 30 "ytsearch:{search_name}" -f 137 --restrict-filenames -o "{target_path}/{dir_name}.mp4"'

        try:
            os.system(command)
        except KeyboardInterrupt:
            print("Caught exception type on main.py ball_detect:",
                    type(KeyboardInterrupt).__name__)
            exit()
        except SystemExit:
            print("Caught exception type on main.py ball_detect:",
                    type(SystemExit).__name__)
        except ValueError:
            print("Caught exception type on main.py ball_detect:",
                    type(ValueError).__name__)
        except ZeroDivisionError:
            print("Caught exception type on main.py ball_detect:",
                    type(ZeroDivisionError).__name__)
        except TypeError:
            print("Caught exception type on main.py ball_detect:",
                    type(TypeError).__name__)
        except IndexError:
            print("Caught exception type on main.py ball_detect:",
                    type(IndexError).__name__)
        except FileNotFoundError:
            print("Caught exception type on main.py ball_detect:",
                    type(FileNotFoundError).__name__)
        except IOError:
            print("Caught exception type on main.py ball_detect:",
                    type(IOError).__name__)
        except OSError:
            print("Caught exception type on main.py ball_detect:",
                    type(OSError).__name__)
        except Exception:
            print("Caught exception type on main.py ball_detect:",
                    type(Exception).__name__)





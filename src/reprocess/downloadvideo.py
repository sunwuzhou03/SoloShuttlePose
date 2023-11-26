import os
import shutil
import yt_dlp

dir_name="Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals"

target_path="ShuttleSet\ShuttleSet22\data4drl\Akane_YAMAGUCHI_AN_Se_Young_BWF_World_Championships_2022_Semi_finals"

search_name=search_name = dir_name.replace("_", " ")

command = f'yt-dlp --write-link --min-sleep-interval 10 --max-sleep-interval 30 "ytsearch:{search_name}" -f 137 --restrict-filenames -o "{target_path}/{dir_name}.mp4"'

os.system(command)

import subprocess
import time
import os

def get_date_from_filename(filename):
    filename = filename.split('.')[0]
    return datetime.datetime.strptime(filename, '%m-%d-%y-%I-%M%p')


for f in os.listdir("raw_videos"):
	subprocess.run("ffmpeg -i raw_videos/{} -map 0 -c copy -f segment -segment_time 3600 videos/{}_%03d.mkv".format(f, f.split('.')[0] ), shell=True)


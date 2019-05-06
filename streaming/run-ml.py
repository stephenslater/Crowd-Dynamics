import subprocess
import glob
import sys
import os
import time

script_loc = "~/Desktop/CS205/Final\ Project/streaming/process_video.py"
model = "faster_rcnn_resnet101_coco_2018_01_28"

while True:
    video_names = os.listdir("videos")
    print('%d videos found' % (len(video_names)))
    for vid in video_names:
        print(vid)
        subprocess.run('python {} --model {} --video {} > logs/{}'.format(script_loc, model, vid, vid), shell=True)
        print("{} Processing complete".format(vid))
        os.remove(os.path.join("videos", vid))
    time.sleep(1)

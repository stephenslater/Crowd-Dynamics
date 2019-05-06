import subprocess
import glob
import sys
import os
import time

script_loc = 'process_video.py' 
model = "faster_rcnn_resnet101_coco"

while True:
    video_names = os.listdir(os.environ['HOME']+"/videos")
    print('%d videos found' % (len(video_names)))
    for vid in video_names:
        if not vid.endswith('.ts'):
            continue
        print(vid)
        print(script_loc)
        subprocess.run('python3 {} --model {} --video {} > logs/{}'.format(script_loc, model, vid, vid), shell=True)
        print("{} Processing complete".format(vid))
        os.remove(os.path.join(os.environ['HOME'] + "/videos", vid))
    time.sleep(1)

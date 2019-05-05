import subprocess
import glob
import sys

day = sys.argv[1]
device = sys.argv[2]

CUDA_CMD = "export CUDA_VISIBLE_DEVICES=%s" % device
script_loc = "Crowd-Dynamics/processing/process_video.py"
model = "faster_rcnn_resnet101_coco_2018_01_28"
video_names = [x.split('/')[1] for x in glob.glob('videos/201904%s*' % day)]
print('%d videos found for day %s. Running on device %s' % (len(video_names), day, device))
for vid in video_names:
    print(vid)
    subprocess.run('{} &&  python {} --model {} --video {} > logs/{}'.format(CUDA_CMD, script_loc, model, vid, vid), shell=True)

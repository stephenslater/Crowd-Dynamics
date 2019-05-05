import subprocess


script_loc = "Crowd-Dynamics/processing/process_video.py"
model = "faster_rcnn_resnet101_coco_2018_01_28"
subprocess.Popen('export CUDA_VISIBLE_DEVICES=0 &&  python {} --model {} --video 160.mkv  > logs/1601.mkv'.format(script_loc, model), shell=True)
subprocess.Popen('export CUDA_VISIBLE_DEVICES=1 &&  python {} --model {} --video min-2.mkv  > logs/1602.mkv'.format(script_loc, model), shell=True)
subprocess.Popen('export CUDA_VISIBLE_DEVICES=2 &&  python {} --model {} --video min-3.mkv  > logs/1603.mkv'.format(script_loc, model), shell=True)
subprocess.Popen('export CUDA_VISIBLE_DEVICES=3 &&  python {} --model {} --video min-4.mkv  > logs/1604.mkv'.format(script_loc, model), shell=True)

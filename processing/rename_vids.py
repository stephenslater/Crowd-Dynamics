import subprocess
import time
import os
import datetime

def get_real_time(filename):
    filename = filename.split('.')[0]
    hour = float(filename.split('_')[1])
    name = filename.split('_')[0]
    time = datetime.timedelta(hours=float(hour)) + datetime.datetime.strptime(name, '%m-%d-%y-%I-%M%p')
    return str(time.strftime("%Y%m%d-%H%M%S")+".mkv")


test = "20190411-141300-12.mkv"

def get_frame_time(filename, frame_number):
	filename = filename.split('.')[0]
	fps  = int(filename.split('-')[2])
	filename = filename[:-3]
	start_time = datetime.datetime.strptime(filename, "%Y%m%d-%H%M%S")
	delta = datetime.timedelta(seconds=fps * frame_number)
	current_time = start_time + delta
	return current_time.strftime("%Y%m%d-%H%M%S")

print(get_frame_time(test, 100))

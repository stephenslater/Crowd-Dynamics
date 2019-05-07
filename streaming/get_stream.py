"""Grabs and downloads chunks of video from a live data source. You can specify
the livestream source using the start, header, and url variables. These chunks
are immediately fed into an object detection model using Tensorflow in another
script. """
import requests
import time
from boto import kinesis
import json
import datetime
import os
import argparse

VIDEO_PATH = os.path.join(os.environ["HOME"], "streaming-videos")
FRAME_RATE = 6

def get_current():
    start = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/chunklist_w1332054195.m3u8"
    r = requests.get(start)
    c = r.content.decode(encoding='utf-8')
    curr_id = (c.split('EXT-X-MEDIA-SEQUENCE:')[1]).split('#')[0]
    return int(curr_id) - 1

if __name__ == "__main__":
    start = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/chunklist_w1332054195.m3u8"

    curr_id = get_current() 
    
    headers={"origin": "https://commonspaces.harvard.edu", "Referer" : "https://commonspaces.harvard.edu/plaza-webcam"}
    max = get_current()
    while True:
        time.sleep(1)
        url = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/media_w1332054195_" + str(curr_id) + ".ts"
        r = requests.get(url)
        if r.status_code != 404:
            video_file = os.path.join(VIDEO_PATH, str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))+ ('-%d.ts' % FRAME_RATE))
            with open(video_file, 'wb') as textfile:
                textfile.write(r.content)
            curr_id += 1
        print(curr_id, r.status_code)


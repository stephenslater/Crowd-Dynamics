import requests
import time
from boto import kinesis
import json

def get_current():
    start = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/chunklist_w1332054195.m3u8"
    r = requests.get(start)
    c = r.content
    id = (c.split('EXT-X-MEDIA-SEQUENCE:')[1]).split('#')[0]
    id = int(id)
    id = id - 1
    return id

if __name__ == "__main__":
    kinesis = kinesis.connect_to_region("us-east-1")
    start = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/chunklist_w1332054195.m3u8"

    id = get_current() 
    #print(r.content)
    

    headers={"origin": "https://commonspaces.harvard.edu", "Referer" : "https://commonspaces.harvard.edu/plaza-webcam"}
    # print(id)
    max = get_current()
    while True:
        time.sleep(2)
        # print(id)
        url = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/media_w1332054195_" + str(id) + ".ts"
        # url = "https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/media_w1332054195_108653.ts"
        # print(url)
        r = requests.get(url)
        print(r.content)
        # if r.status_code == 404:
        #     # print("hello")
        if r.status_code != 404:
            kinesis.put_record("CS205", r.content, "partitionkey")
            with open('videos/movie' + str(id) + ".ts", 'w') as textfile:
                textfile.write(r.content)
            id = id + 1
        print(id)
        # print(r.status_code)

# https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/media_w1332054195_108635.ts
# https://d144v3end3hovo.cloudfront.net/monitor/harvard-spaces/science-center-plaza.stream/media_w1332054195_108624.ts
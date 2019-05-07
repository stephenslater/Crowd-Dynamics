# Streaming

This folder contains all the scripts necessary to do real-time analytics
with a streaming data source.
We ran the streaming analysis on the live webcam overlooking Harvard's Science
Center Plaza.
The data source we used was found
[here](https://commonspaces.harvard.edu/plaza-webcam),
but the code should be able to be adapted for other live video streams.

## Main Components

Detecting objects of interest (people, trucks, food trucks, bikes, etc.) was
done with an object detection model, such as Faster-RCNN.
Then, Spark was used for the actual analytics computations, so we could
process it with high throughput.

1. `get_stream.py` grabs chunks of video from a livestream and downloads them to disk.
2. `process_video.py` uses a deep convnet to process the video chunks into bounding boxes.
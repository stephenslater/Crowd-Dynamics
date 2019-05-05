# Crowd Dynamics at Harvard University

Repository for CS 205 Final Project, Spring 2019

## Overview

The Science Center plaza is a very popular area, home to food trucks, students,
and tourists. The Science Center Plaza also has a live stream video feed 24 
hours per day, which you can check out
[here](https://commonspaces.harvard.edu/plaza-webcam).
However, can we use this data to make decisions, such as when to schedule
events in the plaza? However, we only have a large amount of raw video data,

Our project provides real-time crowd analytics with object detection of people
and average interframe and intraframe metrics, such as number of people, group
size, velocity, and location heatmap for a given time period.
We analyze 700 hours of video and then use streaming to compare the real-time
current analytics to the historical data for the corresponding time period in
order to detect interesting events such as large groups of people or fast
movement.

## Documentation

Our documentation is available on Github Pages.
You can find it
[here.](https://stephenslater.github.io/Crowd-Dynamics/)

## Deployment Steps

### Object Detection

The object detection step is done on Amazon Web Services through EC2, with a
`p3.8xlarge` instance.
We were originally going to use a cluster of 8 `p2.2xlarge` instances, but
after repeated emails with AWS support we were only able to get access
to 1 `p3.8xlarge` instances.
We were still able to take advantage of GPU parallelism using `p3.8xlarge`
instance, which has 4 NVIDIA V100 GPUs.

To replicate the results, download a video from our S3 bucket with video
data and run

TODO: Have RLIN make 

```bash
python run_ml.py
```

Then, the output data

TODO: List where the output data gets sent to

### Crowd Analytics

Once the bounding boxes have been generated using the GPU instance on EC2, we
use a Spark cluster from AWS EMR to run the analytics.
The Spark cluster was created with 8 `m4.xlarge` instances.
Then, we added the detection files to the spark file system on AWS EMR.
We ran the following commands on our spark cluster.

```bash
hadoop fs -put <detection_output>
```

Then, after putting in the path of the detection output into the script,
we can run our analytics with `spark-submit`.
This outputs the computed analytics to a pandas dataframe, which we can then
feed into our visualization code.

```bash
spark-submit heatmap.py
```

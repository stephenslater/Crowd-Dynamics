# Crowd Dynamics at Harvard University

## Problem

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

## Why Big Data? 

This problem requires the use of multiple frameworks:

* __Big Data:__ because we process 150 GB of low-resolution historical video.
* __HPC:__: handling the throughput for streaming analytics of real-time science
center plaza video feed

## Infrastructures and Methodologies

Our solution uses a pre-trained neural network (Faster R-CNN) across 8 GPUs to
identify bounding boxes of people, and then 32 CPU cores to compute the
analytics. For infrastructure, we use AWS EMR, Databricks on Spark ML to run
our TensorFlow model, and Spark Streaming for real-time analytics.

## The Technical Stuff

We used a deep convolutional neural net using the Faster-RCNN architecture [1]
for detecting people, bicycles, cars, and trucks.
We used a pretrained model that was trained on Microsoft COCO [2], a dataset of
common objects in context.
The dataset is composed of a large number of images with a total of 91 unique
objects with labels; however, we only care detecting pedestrians, so we only
focus on detecting one class (person).

## Code and Data

Our code can be found on Github
[here](http://www.github.com/stephenslater/crowd-dynamics).

TODO: Add S3 links to our data files.

## Visualizations

Check out the visualizations
[here!](visualizations.html).
Through the use of deep convolution neural networks and big data platforms
such as Spark, we are able to quantify some of the patterns in the Science
Center.
In the data, we are able to see:

1. Tour groups traveling through the yard.
2. The line for food trucks.
3. The influx of people when classes are dismissed.
4. How long people stay in the science center plaza to eat.
5. and many more!

## Citations

1. Ren, Shaoqing et al. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” IEEE Transactions on Pattern Analysis and Machine Intelligence 39.6 (2017): 1137–1149. Crossref. Web.

2. Lin, Tsung-Yi et al. “Microsoft COCO: Common Objects in Context.” Lecture Notes in Computer Science (2014): 740–755. Crossref. Web.

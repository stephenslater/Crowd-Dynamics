---
title: Overview
---

## Problem

The Science Center plaza is a very popular area, home to food trucks, students,
and tourists. The Science Center Plaza also has a live stream video feed 24 
hours per day, which you can check out
[here](https://commonspaces.harvard.edu/plaza-webcam).
However, can we use this data to make decisions, such as when to schedule
events in the plaza?

Our project provides real-time crowd analytics with object detection of people
and average interframe and intraframe metrics, such as number of people, group
size, velocity, and location heatmap for a given time period.
We analyze 700 hours of video and then use streaming to compare the real-time
current analytics to the historical data for the corresponding time period in
order to detect interesting events such as large groups of people or fast
movement.

## Big Data and Big Compute

This problem requires the use of multiple frameworks:

* __Big Data:__ because we process 150 GB of low-resolution historical video.
* __Big Compute/HPC:__ handling the throughput for streaming analytics of real-time science
center plaza video feed

With the Big Data part of our solution, we have fine-grained data parallelism and a SPMD execution model,
while with the Big Compute component, we have coarse-grained data and functional parallelism and a DAT
execution model.

## Data, Software, and Infrastructure

We continuously collected from the Science Center Plaza Webcam over the course of multiple weeks.

The basic outline of our solutions is as follows. We begin by feeding our data through an object detection
model to obtain bounding boxes identifying person locations. We then compute analytics such as number of
people, velocities, gorup sizes, and locations based on bounding boxes. For historical analysis, we then
perform some aggregation of these statistics over short periods of time. For streaming, we just directly work
with the output analytics. After analytics computation is complete, we feed the results into a variety of visualizations.

We use AWS EC2 GPU instances to run object detection, and Spark on AWS EMR to perform analytics computation.

See the [implementation page](implementation.html) for more details on our software and infrastructure.

## Code and Data

Our code can be found on Github
[here](http://www.github.com/stephenslater/crowd-dynamics).

Our data can be found on S3:
    * Processed Bounding Boxes: [here](https://s3.amazonaws.com/science-center-plaza-data/data/all_detections.tar.gz)
    * Processed Bounding Boxes Combined: [here](https://s3.amazonaws.com/science-center-plaza-data/data/all_detections_combined.tar.gz)
    * Video Data: [here](https://s3.console.aws.amazon.com/s3/buckets/science-center-plaza-data)

## Visualizations

<!-- <div style="width:100%; background-color:red; height: 308px"> -->

<iframe src="https://giphy.com/embed/cms6JM0agpP9HfWIEy" width="720" height="462" frameBorder="0" class="giphy-embed" style="display: block; margin-left: auto; margin-right: auto;width: 720px" allowFullScreen></iframe>

<!-- </div> -->

Check out more visualizations [here!](visualizations.html)
Through the use of deep convolution neural networks and big data platforms
such as Spark, we are able to quantify some of the patterns in the Science
Center.
In the data, we are able to see:

<ul>
    <li>Tour groups traveling through the yard.</li>
    <li>The line for food trucks.</li>
    <li>The influx of people when classes are dismissed.</li>
    <li>How long people stay in the science center plaza to eat.</li>
    <li>and many more!</li>
</ul>

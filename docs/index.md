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

We continuously collected from the Science Center Plaza Webcam over the course of multiple weeks from 4/5/19 to 5/3/19.

The basic outline of our solutions is as follows. We begin by feeding our data through an object detection
model to obtain bounding boxes identifying person locations. We then compute analytics such as number of
people, velocities, group sizes, and locations based on bounding boxes. For historical analysis, we then
perform some aggregation of these statistics over short periods of time. For streaming, we just directly work
with the output analytics. After analytics computation is complete, we feed the results into a variety of visualizations.

Note that our basic features are use of Spark and GPU, and our advanced feature is use of TensorFlow for our ML model.
Our advanced functionality, we had scheduling of deep learning jobs across 4 GPUs, an alert system comparing real-time 
analytics to historical data for the corresponding hour, and a variety of visualizations including interactive plots.

We use AWS EC2 GPU instances to run object detection, and Spark on AWS EMR to perform analytics computation.

See the [implementation page](implementation.html) for more details on our software and infrastructure.

## Code and Data

Our code can be found on Github
[here](http://www.github.com/stephenslater/crowd-dynamics).

Our data can be found on S3:

* Processed Bounding Boxes: [here](https://drive.google.com/open?id=1g1MuCQdZyXJoDIY28-wriIBqh4G_DRgT)

* Processed Bounding Boxes Combined: [here](https://drive.google.com/open?id=1uLOzrqadUjTC3b4PvPaHHIJkFJY0azRW)

* Video Data: [here](https://drive.google.com/drive/folders/15Ui7FiJQtIAhgsCwmYHJ0k8dIodDR7x5?usp=sharing)


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

## Conclusions

Overall, we successfully used big data and big compute infrastructures combined with TensorFlow to develop analytics of 
real-time crowd dynamics compared to historical data for the Science Center Plaza.


We ran into a few challenges in the course of our work. First, Databricks was unhelpful in distributing TensorFlow model 
across GPUs, so we ended up using an AWS EC2 GPU instance (p3.8xlarge) with 4 GPUs for our object detection model
evaluation instead. We also tried to do some data processing and cleaning in PySpark before computation, but the API 
was severely limited due to distribution requirements (i.e. difficult to associate data across columns, create arbitrary data, 
and work with multi-dimensional arrays). Our solution to this was to perform some preprocessing in plain Python. We discovered
empirically that UDFs with 2-column inputs would not parallelize well (the cluster was reverting to using only 1 core on 1 
executor), so we concatenated and flattened bounding box data for adjacent frames to compute successive-frame statistics
(e.g., velocity) using a single row.


From these challenges, we learned that data format needs to be very carefully selected before loading it into a Spark 
Dataframe, and we need to be careful when developing how to compute analytics, and which metrics to compute in a distributed 
system (i.e. choice of UDF).


In the future, we might want to handle more than one camera for the streaming (if there were more cameras) and take advantage 
of GPU IO system with AWS (NV-Link).

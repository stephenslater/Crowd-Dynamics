---
title: Implementation
---
Here we describe the infrastructure, technologies, and platforms we used
throughout the entire pipeline.

## Data Pipeline

We mainly use distributed Tensorflow for the machine learning inference and
Spark for the following data analytics.
Our pipeline looks like this:

![implementation](images/pipeline.png)

Our project can be split up into multiple stages.

### Video Processing and Object Detection

First, we read in the historical video records of the Science Center Plaza and turn our collected video into 1 hour chunks.

We then use a TensorFlow implementation of an object detection model to generate bounding boxes.
In particular, we use a deep convolutional neural net using the Faster-RCNN architecture [1]
for detecting people, bicycles, cars, and trucks. We used a pretrained model that was trained on Microsoft COCO [2], 
a dataset of common objects in context. The dataset is composed of a large number of images with a total of 91 unique
objects with labels; however, we only care detecting pedestrians, so we only focus on detecting one class (person). 
For each frame, we end up generating a timestamp, bounding boxes, and scores for confidence of detection in each of those 
bounding boxes. We save this output data to Spark dataframes, where each row contains the output generated from one frame, which are then used when we compute our analytics.

We run this stage of our computation on a AWS EC2 instance with GPU, in particular a p3.8xlarge instance, which has 4 GPUs
with a Deep Learning AMI (Ubuntu) Version 22.0.

### Analytics

To perform our analytics, we apply user defined function transformations to the columns of the Spark dataframes from the 
object detection step. These analytics include locations of people, number of people, group size, and velocities. After 
computing analytics on the historical data, we also aggregate over short windows of time (ex. 10 minutes). After all analytics
computations have been completed, we write our results to a Pandas dataframe to be used in the visualization code.

We run this stage of our computation on a AWS EMR cluster with Spark. In particular, we use emr-5.23.0, which has
Spark 2.4.0 on Hadoop 2.8.5 YARN with Ganglia 3.7.2 and Zeppelin 0.8.1, and have 8 m4.xlarge instances in our cluster.

We now provide some more details on how our analytics transformation were done. Locations of people and number of people are both fairly straightforward, to calculate, with location computed by finding the centers of the input bounding boxes and
number of people found by just counting the number of bounding boxes. Group size and velocity computations are a bit more involved, however, and are described in detail below.

#### Group Size

TODO: Add details of how this is done.

#### Velocity

In order to compute the average velocity in each frame, we must consider the detected objects in the next frame.
For person A in frame $$i$$, we need to identify the location of person A in frame $$i+1$$, and then approximate the velocity in frame i by dividing the distance traveled (between person A's centers) by the time, which is the inverse of the fps (frames per second).

So, first, we must link people between frames.
Since people are constantly entering and leaving the frame, the number of objects per frame ($n_i$) may not be consistent between successive frames.
Therefore, for each frame, we can compute at most $$\text{min}(n_i, n_{i+1})$$ velocities.
We compute pairwise distances between all objects in different frames, sort the pairs in ascending order of distance, and then greedily label each unassigned object in frame $$i$$ to the closest unassigned object in frame $$i+1$$.
We account for the edge case of a person leaving one end of the frame and another person entering on the opposite side of the next frame by only linking people whose distance is within a predefined threshold of 30% of the frame width or length.
In particular, we compute the Euclidean distance between the centers (where the two dimensions of the centers are in $$[0, 1]$$ as proportions of the frame dimension length), and then do not link objects if the distance exceeds $$0.3$$.

Here is an example image, where the first frame (A) has 6 detected people, and the second frame (B) has 3 detected people. The green lines denote the link, and the velocity is computed as the length of the green line times the fps.

<p align="center"> 
<img src="images/linking.png">
</p>

### Visualizations
We take the augmented Spark dataframe from our analytics step and convert it back into a Pandas dataframe for
visualization. With this dataframe, create visualizations using the Bokeh library for interactive graphs,
and CV2 with matplotlib to draw the bounding boxes on the original video.

TODO: Add a more detail.

## Citations

1. Ren, Shaoqing et al. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” IEEE Transactions on Pattern Analysis and Machine Intelligence 39.6 (2017): 1137–1149. Crossref. Web.

2. Lin, Tsung-Yi et al. “Microsoft COCO: Common Objects in Context.” Lecture Notes in Computer Science (2014): 740–755. Crossref. Web.

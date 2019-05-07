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
bounding boxes. We save this output data to Spark dataframes, which are then used when we compute our analytics.

We run this stage of our computation on a AWS EC2 instance with GPU, in particular a p3.8xlarge instance, which has 4 GPUs
with a Deep Learning AMI (Ubuntu) Version 22.0.

### Analytics

To perform our analytics, we apply user defined function transformations to the columns of the Spark dataframes from the object detection step. We use these functions to create new columns with individualized and aggregated metrics 

TODO: Add a lot more detail here.

We run this stage of our computation on a AWS EMR cluster with Spark. In particular, we use emr-5.23.0, which has
Spark 2.4.0 on Hadoop 2.8.5 YARN with Ganglia 3.7.2 and Zeppelin 0.8.1, and have 8 m4.xlarge instances in our cluster.

### Visualizations
We take the augmented Spark dataframe from our analytics step and convert it back into a Pandas dataframe for
visualization. With this dataframe, create visualizations using the Bokeh library for interactive graphs,
and CV2 with matplotlib to draw the bounding boxes on the original video.

TODO: Add a more detail.


## Citations

1. Ren, Shaoqing et al. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” IEEE Transactions on Pattern Analysis and Machine Intelligence 39.6 (2017): 1137–1149. Crossref. Web.

2. Lin, Tsung-Yi et al. “Microsoft COCO: Common Objects in Context.” Lecture Notes in Computer Science (2014): 740–755. Crossref. Web.

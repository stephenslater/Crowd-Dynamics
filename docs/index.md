# Crowd Dynamics at Harvard University

## Problem

The Science Center plaza is a very popular area.

TODO: Fill out problem

## Solution

## The Technical Stuff

We used a deep convolutional neural net using the Faster-RCNN architecture [1] for detecting people, bicycles, cars, and trucks.
We used a pretrained model that was trained on Microsoft COCO [2], a dataset of common objects in context.
The dataset is composed of a large number of images with a total of 91 unique objects with labels; however, we only care detecting pedestrians,
so we only focus on detecting one class (person).

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

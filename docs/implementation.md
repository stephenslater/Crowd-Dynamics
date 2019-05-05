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

First, we read in the historical video records of the Science Center Plaza.
We then use Tensorflow for our machine learning to generate bounding boxes
around objects of interest, including people, food trucks, bicycles, and cars.
We used the Tensorflow framework, and used one `p3.8xlarge` instance (which has 4 GPUs)
to generate bounding boxes and class scores. Bounding boxes and scores are output into Spark 
dataframes, which is then used to run our analytics.

### Analytics

TODO: Write up analytics section

### Visualizations

TODO: Write up visualizations section

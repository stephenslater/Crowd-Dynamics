# Platform and Infrastructure

## Data Pipeline

We mainly use distributed Tensorflow for the machine learning inference and
Spark for the following data analytics.
Our pipeline looks like this:

![implementation](images/pipeline.png)

Our project can be split up into multiple stages.

### Video Processing

First, we process the historical video records of the Science Center Plaza.
We used 144 cores to turn the historical video into individual frames,
encoded losslessly as `.png` files.
These individual `.png` frames are then stored into the AWS EMRFS, which is the
filesystem associated with our AWS EMR cluster. 

### Bounding Boxes

We then use Tensorflow for our machine learning to generate bounding boxes
around objects of interest, including people, food trucks, bicycles, and cars.
We used the Tensorflow framework, and used between one to eight `p2.xlarge`
instances to generate the bounding boxes.
The bounding boxes are outputted into a Spark Dataframe, which is then used
to run our analytics.

### Analytics

TODO: Write up analytics section

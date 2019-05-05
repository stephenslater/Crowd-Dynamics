# Tuning our Pipeline

## Object Detection Model Selection

One of the most important parts of our pipeline is the object detection model
that produces bounding boxes for our further analysis.
The Science Center plaza webcam has no labels, and it would be infeasible
for us to provide ground truth labels for this dataset.
Therefore, we used a pretrained model from the Microsoft COCO dataset.
The Microsoft COCO dataset has labeled images of people, trucks, bicycles,
cars, and many more relevant classes - however, the images are usually from
a different perspective and very different distance.
Therefore, we cannot use the reported mAP for these models, and will have
to evaluate the performance of these models by eye.
We looked at the archiectures:

* Faster-RCNN
* SSD
* YOLO

We also looked at the following pretty standard feature extractors:

* Resnet-50
* Resnet-101
* Inception-V2

We can evaluate these pretained models side by side.

### Comparisons of Different Models

There is a tradeoff between how long it takes for inference and the quality
of the predictions.
We evaluate the model's performance during different times of the day.

#### Daytime, Not Crowded

<html>
 <body>
<iframe src="http://www.youtube.com/embed/F08-z8duKIE"
   width="560" height="315" frameborder="0" allowfullscreen></iframe>
 </body>
</html>

#### Slightly Darker, Not Crowded

<html>
 <body>
<iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
 </body>
</html>

<html>
 <body>
  <object data="http://www.youtube.com/embed/dQw4w9WgXcQ"
   width="560" height="315"></object>
 </body>
</html>

#### Nighttime, around 8 PM

<html>
 <body>
  <object data="http://www.youtube.com/embed/W7qWa52k-nE"
   width="560" height="315"></object>
 </body>
</html>
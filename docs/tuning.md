---
title: Tuning
---
## Object Detection Model Selection

One of the most important parts of our pipeline is the object detection model
that produces bounding boxes for our further analysis.
The Science Center plaza webcam has no labels, and it would be infeasible
for us to provide ground truth labels for this dataset.
Therefore, we used a pretrained model from the Microsoft COCO dataset.
The Microsoft COCO dataset has labeled images of people, trucks, bicycles,
cars, and many more relevant classes--however, the images are usually from
a different perspective and very different distance.
Therefore, we cannot use the reported mAP (mean Average Precision) for these models, and will have
to evaluate the performance of these models visually.
We looked at the architectures:

* Faster-RCNN
* SSD
* YOLO

We also looked at the following standard feature extractors:

* Resnet-50
* Resnet-101
* Inception-V2

We can evaluate these pretained models side by side.

### Comparisons of Different Models

There is a tradeoff between how long it takes for inference and the quality
of the predictions. The model with the best quality (Faster-RCNN with Resnet-101) has the slowest inference time. We chose to use this model due to the best performance in object detection.

We evaluate the model's performance during different times of the day.

#### Daytime, Not Crowded

<iframe width="560" height="315" src="https://www.youtube.com/embed/F08-z8duKIE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen class="youtube"></iframe>

#### Slightly Darker, Not Crowded

<iframe width="560" height="315" src="https://www.youtube.com/embed/ZR53NL4JOVU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen class="youtube"></iframe>

#### Nighttime, around 8 PM

<iframe width="560" height="315" src="https://www.youtube.com/embed/KnjFIt1sypg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen class="youtube"></iframe>

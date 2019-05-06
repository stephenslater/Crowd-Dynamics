---
title: Visualizations
---
Here are some visualizations of what the detections look like.

## Hotspots

How does the distribution of people in the plaza change over time?
Check out the heatmap of where people are standing in the image here.

<link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.12.3.min.css" type="text/css" />
        
<script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.12.3.min.js"></script>

<iframe src="locations.html"
        sandbox="allow-same-origin allow-scripts"
        width="500"
        height="550"
        scrolling="no"
        seamless="seamless"
        frameborder="0" style="display: block; margin-left: auto; margin-right: auto; width: 500">
</iframe>

## Plaza Metrics

### Average Number of People in the Plaza

Ever wonder when people are around the plaza the most?
We're able to see when people start moving around the square, and how many
people are moving at any point in time.
We count the number of people in the square at any point in time.
For example, here is a bar chart for the average number of people in the plaza,
while aggregating with a window of 20 minutes for Thursday, April 11.
We are able to see from the aggregated analytics when classes finish.

![classtimesthursday](images/classtimesthursday.png)

If you want to explore a subset of the data yourself, check out 
[this interactive graph.](interactive.md)
You can hover over the bars to see more information and zoom in.
Not all of the data can be loaded because there is too much data and it lags
out the webpage.

## Looking at the Detections

It's also nice to be able to look at the detections themselves.
We are able to get the detections and plot different classes to check if the
detections are reasonable.

<iframe width="560" height="315" src="https://www.youtube.com/embed/eN9tTVJ9J2c" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen class="youtube"></iframe>


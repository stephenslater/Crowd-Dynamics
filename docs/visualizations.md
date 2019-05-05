# Visualizations

Here are some visualizations of what the detections look like.

## Hotspots

How does the distribution of people in the plaza change over time?
Check out the heatmap of where people are standing in the image here.

<link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.12.3.min.css" type="text/css" />
        
<script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.12.3.min.js"></script>

<iframe src="locations.html"
        sandbox="allow-same-origin allow-scripts"
        width="100%"
        height="550"
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>

## Average Velocity Distribution

Ever wonder when people are moving around the plaza the most?
We're able to see when people start moving around the square, and how many
people are moving at any point in time.

<iframe src="synthetic_velocities.html"
    sandbox="allow-same-origin allow-scripts"
    width="100%"
    height="550"
    scrolling="no"
    seamless="seamless"
    frameborder="0">
</iframe>

## Looking at the Detections

It's also nice to be able to look at the detections themselves.
We are able to get the detections and plot different classes to check if the
detections are reasonable.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Q-dZt2IdMKg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

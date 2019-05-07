# Computing Analytics with Spark

We used Spark to handle the analytics after we get the object detection
bounding boxes from our deep convolutional neural network.
We compute a few values aggregated over a time window, usually around 5 minutes
to 30 minutes depending on how fine-grained you want the analytics to be.
Here are some of the possible things to calculate:

* Average Group Size
* Average Velocity
* Location of people within a supplied time window
* Average Number of people in a time window
* etc.

When using a fine-grained aggregation window of 5 minutes, you can clearly
see what paths across the plaza are popular at certain times (people like
traveling from the Oxford Street to the west gate to the yard at around 5PM
to 6PM, which makes sense since that's when a lot of the graduate students
leave the science center region).
Using a more coarse-grained aggregation window gives the advantage of easily
being able to compare statistics across days to see if a day stands out (for
example, the heatmap and average groupsizes were a lot different during
the Arts First event, which had many events in the plaza).

## Scripts

The scripts were all run in an EMR cluster. We used the latest version of EMR
with Spark 2.4.0, and ran everything using `m4.xlarge` instances.

1. `analytics.py`: Reads in object detections from the processing scripts and computes analytics. Returns a `.csv` with the aggregated analytics. You will need to use `spark-submit` to run this.
2. `bench.py`: Benchmarking script to evaluate the speedup when tuning the number of cores and executors.
3. `bench_spark.py`: Helper script that is used during the benchmarking process.
4. `heatmap.py`: Reads in object detections from the processing scripts and performs an aggregation that allows us to plot a heatmap of where people go over time windows. This is separated out because the aggregation process doesn't reduce the amount of data much compared to the other metrics such as average group size. Must be run with `spark-submit`.

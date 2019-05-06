# PySpark Analytics for Crowd Dynamics
# Harvard University
# CS 205, Spring 2019
# Group 1

# Libraries
import pandas as pd
import numpy as np
from functools import reduce

# PySpark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.sql.functions import udf, array, avg, col, size, struct, lag, window, countDistinct, monotonically_increasing_id, collect_list
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, ArrayType, LongType, FloatType
from pyspark.sql.window import Window

# Window size
window_minutes = 20. 
fps = 2.

spark = SparkSession.builder.getOrCreate()

"""# Load data"""
directory = 'folder'
df = spark.read.json(directory)
df = df.orderBy('timestamp').show()

"""# Functions used for UDFs (including velocity and group size)"""
def count(column):
    return len(column)
  
def sum_vals(column):
    return float(sum(column))
  
def avg_vals(columns):
    return float(columns[0] / columns[1]) if columns[1] else 0.0
  
def get_center(values):
    res = []
    for i in range(int(len(values)/4)):
        y1, x1, y2, x2 = values[4*i:4*i+4]
        x_mean, y_mean = round(float(x1+x2)/2, 3), round(float(y1+y2)/2, 3)
        res.extend([x_mean, y_mean])
    return res

def get_x(values):
    return [values[i] for i in range(0, len(values), 2)]

def get_y(values):
    return [values[i] for i in range(1, len(values), 2)]
  
def fudf(val):
    return reduce(lambda x, y:x+y, val)

def compute_velocities(cols, fps=2.0, threshold=0.3, return_assignments=False):
    # Assume frame_1 = [x_i, y_i, x_{i+1}, y_{i+1}, ...]
    f_1 = iter(cols[0])
    f_2 = iter(cols[1])
    frame_1 = list(zip(f_1, f_1))
    frame_2 = list(zip(f_2, f_2))
    
    val_1 = {k: v for k, v in enumerate(frame_1)}
    val_2 = {k: v for k, v in enumerate(frame_2)}

    # Compute pairwise distances
    distances = {}
    for i in range(len(frame_1)):
        for j in range(len(frame_2)):
            # Euclidean distance between two people
            distances[i, j] = np.sqrt(
                (val_1[i][0] - val_2[j][0]) ** 2 + (val_1[i][1] - val_2[j][1]) ** 2)

    # Assigned ids from frame 1 (reference frame), {id_i: vel_i}
    velocities = dict()

    # Assigned ids from frame 2 (with values as match in frame 1)
    targets = dict()
    num_assigned = 0
    num_ids = min(len(frame_1), len(frame_2))

    # Sort distances by key: (id in frame 1, id in frame 2)
    pairs = sorted(distances.items(), key=lambda v:v[1])
    for p, dist in pairs:
        # Stop assigning ids when the distance exceeds a user-defined threshold
        # i.e. this covers the case when a person leaves one end of the image
        # and another person enters at the opposite side. We should not match
        # these ids to each other.
        if dist > threshold:
            break

        # Found closest ids between frames
        if p[0] not in velocities and p[1] not in targets and num_assigned < num_ids:
            num_assigned += 1
            # Velocity (distance units per second)
            velocities[p[0]] = dist * fps
            targets[p[1]] = p[0]

    if return_assignments:
        assignments = [[v, k] for k, v in targets.items()]
        return [float(v) for v in velocities.values()], assignments
    return [float(v) for v in velocities.values()]

# DFS to find connected components, where edges connect i, j iff dist(i,j)<threshold 
def dfs_all(graph):
    def dfs(node, graph):
        stack = [node]
        cc = [node]
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    cc.append(v)
                    stack.append(v)
        return cc

    ccs = []
    visited = [False for _ in range(len(graph))]
    for i in range(len(graph)):
        if not visited[i]:
            visited[i] = True
            cc = dfs(i, graph)
            ccs.append(cc)
    return list(map(len, ccs))

# Find groups of detected people by constructing a graph
def compute_groups(positions, threshold=0.1):
    p_1 = iter(positions)
    positions = list(zip(p_1, p_1))
                   
    # Compute pairwise distances
    graph = {i: set() for i in range(len(positions))}
    for i in range(len(positions)):
        for j in range(i, len(positions)):
            # Euclidean distance between two people
            dist = np.sqrt(
                (positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2)
            # Add edge to graph 
            if dist < threshold:
                graph[i].add(j)
                graph[j].add(i)
    lengths = dfs_all(graph)           
    return lengths

"""# UDFs"""

count_udf = udf(count, IntegerType())
sum_udf = udf(sum_vals, DoubleType())
avg_udf = udf(lambda arr: avg_vals(arr), DoubleType())
center_udf = udf(get_center, ArrayType(FloatType()))
velocity_udf = udf(lambda arr: compute_velocities(arr), ArrayType(DoubleType()))
group_udf = udf(compute_groups, ArrayType(IntegerType()))
x_udf = udf(get_x, ArrayType(DoubleType()))
y_udf = udf(get_y, ArrayType(DoubleType()))
flattenUdf = udf(fudf, ArrayType(DoubleType()))

"""# Window size for pairwise shifting"""

w_pair = Window().partitionBy().orderBy(col("timestamp"))

"""# Create and Modify Columns"""

df = (df.withColumn('num_people', count_udf('scores'))
        .withColumn('centers', center_udf('bboxes'))
        .withColumn('x_centers', x_udf('centers'))
        .withColumn('y_centers', y_udf('centers'))
        .withColumn('group_sizes', group_udf('centers'))
        .withColumn('num_groups', count_udf('group_sizes'))
        .withColumn('next_frame_centers', lag("centers", -1).over(w_pair)).na.drop()
        .withColumn('velocities', velocity_udf(struct('centers', 'next_frame_centers')))
        .withColumn('num_velocities', count_udf('velocities'))
        .withColumn('sum_velocities', sum_udf('velocities')))
        
df.show()

"""# Aggregate each 5 minute window to compute:
- average number of people detected
- average group size
- average velocity
"""

# seconds = window_minutes * 60
window_str = '{} minutes'.format(window_minutes)
agg_df = (df.groupBy(window('timestamp', windowDuration=window_str, slideDuration=window_str))
           .agg(F.sum('num_people'),
                F.sum('num_groups'),
                F.sum('sum_velocities'),
                F.sum('num_velocities'),
                avg('num_people'),
                collect_list('x_centers'),
                collect_list('y_centers'))
           .withColumn('avg_group_size', avg_udf(struct('sum(num_people)', 'sum(num_groups)')))
           .withColumn('avg_velocity', avg_udf(struct('sum(sum_velocities)', 'sum(num_velocities)')))
           .withColumnRenamed('avg(num_people)', 'avg_num_people')
           .withColumn('x_centers', flattenUdf('collect_list(x_centers)'))
           .withColumn('y_centers', flattenUdf('collect_list(y_centers)'))
           .drop('collect_list(x_centers)')
           .drop('collect_list(y_centers)')
           .drop('sum(num_people)')
           .drop('sum(num_groups)')
           .drop('sum(sum_velocities)')
           .drop('sum(num_velocities)')
           .orderBy('window'))

agg_df.show()
pandas_df = agg_df.toPandas()
pandas_df.to_csv('{}-{}mins.csv'.format(directory, window_minutes))

import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
import datetime
import tqdm
import numpy as np
from pathlib import Path
import argparse
import cv2
from pyspark.sql.types import *
from pyspark.sql import SparkSession

sess = tf.Session()
graph = tf.get_default_graph()

MODEL_PATH = os.path.join(os.environ['HOME'], "models")
VIDEO_PATH = os.path.join(os.environ['HOME'], "streaming-videos")
OUTPUT_PATH = os.path.join(os.environ['HOME'], "streaming-output")

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, dest='model', type=str,
                    help="Name of model in model folder")
args = parser.parse_args()


CV_MODEL = args.model
saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, CV_MODEL, 'model.ckpt.meta'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(MODEL_PATH, CV_MODEL)))

input_tensor = graph.get_tensor_by_name('image_tensor:0')
output_tensors = dict(
    bboxes = graph.get_tensor_by_name('detection_boxes:0'),
    classes = graph.get_tensor_by_name('detection_classes:0'),
    n = graph.get_tensor_by_name('num_detections:0'),
    scores = graph.get_tensor_by_name('detection_scores:0'),
)

def pred_from_frame(frames):
    """Takes a list of frames and runs it through our prediction"""
    frame = np.stack(frames)
    output = sess.run(output_tensors, 
         feed_dict={input_tensor: frame})
    bboxes, scores, n, classes = output['bboxes'], output['scores'], output['n'], output['classes']
    return bboxes, scores, n, classes

def compute_center(bbox):
    y1, x1, y2, x2 = bbox
    return round(float(x1+x2)/2, 3), round(float(y1+y2)/2, 3)

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

def compute_groups(positions, threshold=0.1):
    # Compute pairwise distances
    graph = {i: set() for i in range(len(positions))}
    for i in range(len(positions)):
        for j in range(i, len(positions)):
            # Euclidean distance between two people
            dist = np.sqrt((positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2)
            # Add edge to graph 
            if dist < threshold:
                graph[i].add(j)
                graph[j].add(i)
    lengths = dfs_all(graph)           
    return lengths

def compute_velocities(paired_centers, fps=2.0, threshold=0.3, return_assignments=False):
    frame_1, frame_2 = paired_centers
    
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

def compute_avg(vals):
    return sum(vals) / len(vals) if len(vals) else 0.

def process_video(video_path, batch_size=32, rate=1):
    split_name = os.path.splitext(os.path.basename(video_path))[0].split('-')
    timestamp = '-'.join(split_name[:-1])
    fps = int(split_name[-1])
    skip = int(fps // rate)
    initial = datetime.datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
        
    cap = cv2.VideoCapture(video_path)
    all_scores, all_classes, all_n, all_bboxes, timestamps  = [], [], [], [], []
    start_time = time.time()
    video_running = True
    total_frames = 0
    processed = 0
    while video_running:
        frames = []
        for _ in range(batch_size):
            for _ in range(skip):
                ret, frame = cap.read()
                if not ret:
                    video_running = False
                    break 
                total_frames += 1
            if not video_running:
                break
            frames.append(frame)
            processed += 1
            timestamps.append(str(initial + datetime.timedelta(seconds=total_frames/fps)))
        if not frames:
            break
        bboxes, scores, n, classes = pred_from_frame(frames)
        all_scores.append(scores)
        all_bboxes.append(bboxes)
        all_n.append(n)
        all_classes.append(classes)
        if not video_running:
            break
    print('Total frames: %d, frames processed: %d' % (total_frames, processed))
    print("ML time: {} seconds".format(int(time.time() - start_time)))
    full_bboxes = np.row_stack(all_bboxes)
    full_scores = np.row_stack(all_scores)
    full_classes = np.row_stack(all_classes)
    full_n = np.concatenate(all_n, axis=None)
    return timestamps, full_bboxes, full_scores, full_n, full_classes

def make_predictions(videoname):
    video = os.path.join(VIDEO_PATH, videoname)
    BATCH_SIZE = 48
    RATE = 6 
    start_time = time.time()
    timestamps, bboxes, scores, n, classes = process_video(video, batch_size=BATCH_SIZE, rate=RATE)
    
    # some cleaning
    agg_bboxes = []
    num_frames = len(bboxes) 
    for ind in range(num_frames):
        image_n = int(n[ind])
        image_bboxes = bboxes[ind][:image_n]
        image_classes = classes[ind][:image_n]
        image_bboxes = image_bboxes[image_classes == 1.]
        agg_bboxes.append(image_bboxes.tolist())
    
    num_detections = [float(len(x)) for x in agg_bboxes][:-1]
    agg_bboxes = agg_bboxes[:-1]
    centers = [list(map(compute_center, bboxes)) for bboxes in agg_bboxes] 
    pair_centers = list(zip(*(centers[:-1], centers[1:])))
    group_size = list(map(compute_groups, centers))
    velocities = list(map(compute_velocities, pair_centers))      
    avg_group_size = list(map(compute_avg, group_size))
    avg_velocities = list(map(compute_avg, velocities))   
    
    # TODO: Add code to visualize/otherwise use above computed values here
    print ("Time of frame 1: {}\nTime of frame {}: {}".format(timestamps[0], len(timestamps), timestamps[-1]))

    
    end_time = time.time()
    print("Total time: {} seconds".format(time.time() - start_time))
    
            
while True:
    videos = sorted(os.listdir(VIDEO_PATH))
    if not videos:
        time.sleep(1)
    for filename in videos:
        print('Processing video %s' % filename)
        make_predictions(filename)
        os.remove(os.path.join(VIDEO_PATH, filename))

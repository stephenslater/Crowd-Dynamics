import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
import tqdm
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import cv2
import copy
import mss


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

if __name__ == '__main__':
    sess = tf.Session()
    graph = tf.get_default_graph()

    MODEL_PATH = os.path.join(os.environ['HOME'], "models")
    
    # Alert if frame metric exceeds (THRESHOLD * historical average)
    # Columns of history are: [avg group size, avg velocity, avg num people]
    # Rows are 0, ..., 23 for the start of each of the 24 hours per day
    THRESHOLD = 1.3
    history = np.load('frame_averages.npz')['data']
    thresholds = THRESHOLD * history
    alert_msg = ['Average group size is high!', 'Average velocity is high!',
                 'Number of people is high!']
    alert_color = (0, 0, 255)
    # This position should be calibrated to the location of the plaza webcam screencap
    start_height = 55
    diff = 40
    height = [start_height + i * diff for i in range(3)]

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str,
                        default='faster_rcnn_resnet101_coco_2018_01_28',
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

    SCALING = 1.2
    monitor = {'top': 160,
            'left': 160,
            'width': int(640 * SCALING),
            'height': int(360 * SCALING)}
    sct = mss.mss()
    prev_centers = None
    while True:
        last_time = time.time()
        grabbed_image = sct.grab(monitor)
        image = np.array(grabbed_image)
        # Need to convert BGRA to BGR
        image = image[:,:,:3]

        # Run this image through the ML model.
        frame = np.expand_dims(image, 0)
        output = sess.run(output_tensors, feed_dict={input_tensor: frame})
        bboxes, scores, n, classes = output['bboxes'], output['scores'], output['n'], output['classes']

        # Filter for high scores.
        indices = scores > 0.3
        bboxes = bboxes[indices]
        scores = scores[indices]
        classes = classes[indices]

        indices = classes == 1.
        bboxes = bboxes[indices]
        scores = scores[indices]
        classes = classes[indices]

        num_dets = len(bboxes)
        centers = list(map(compute_center, bboxes))
        group_size = compute_groups(centers)
        avg_group_size = compute_avg(group_size)

        velocities = None
        avg_velocity = None
        # Computing velocities requires two successive frames with an identified person
        if prev_centers:
            velocity = compute_velocities(prev_centers, centers)
            avg_velocity = compute_avg(velocity)
        prev_centers = centers
        
        # Make a copy of the image for displaying.
        display_image = image[:]
        display = np.array(image[:])
        # Scale the bounding boxes to the image size.
        h, w, _ = image.shape
        bboxes *= np.array([h, w, h, w])
        num_dets = len(bboxes)
        if num_dets > 0:
            centers_y = bboxes[:, [0, 2]].mean(axis=1)
            centers_x = bboxes[:, [1, 3]].mean(axis=1)
            for i in range(num_dets):
                cx, cy = centers_x[i], centers_y[i]
                color = (0, 255, 0) if classes[i] == 1 else (0, 0, 255)
                cv2.circle(display, (cx, cy), 5, color, -1)
        
        msg = "Number of people: {}".format(num_dets)
        vel_msg = "Average velocity: {}".format(avg_velocity)
        gp_msg = "Average group size: {}".format(avg_group_size)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontcolor = (0, 0, 0)
        fontscale = 1
        cv2.putText(display, msg, (10, 420), font, fontscale, fontcolor, 2, cv2.LINE_AA)
        cv2.putText(display, vel_msg, (10, 380), font, fontscale, fontcolor, 2, cv2.LINE_AA)
        cv2.putText(display, gp_msg, (10, 340), font, fontscale, fontcolor, 2, cv2.LINE_AA)
 
        # Compare current frame to historical average for corresponding hour
        stats = np.array([avg_group_size, avg_velocity, num_dets])
        # alerts = stats >= thresholds[hour]
        alerts = [True, True, True] # Testing
        
        if np.any(alerts):
            print ("\n*\n*\n*\n*\n*ALERT!")
            for i, alert in enumerate(alerts):
                if alert:
                    cv2.putText(display, alert_msg[i], (10, height[i]), font, fontscale, alert_color, 
                                2, cv2.LINE_AA)
                print ("{}: {}".format(alert_msg[i], stats[i]))
            print ("\n*\n*\n*\n*\n*")
        
        cv2.imshow("Science Center Plaza Stream", display)
        print("fps: {}".format(1 / (time.time() - last_time)))
        
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

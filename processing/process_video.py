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

OUTPUT_PATH = os.path.join(os.environ['HOME'], "output")
MODEL_PATH = os.path.join(os.environ['HOME'], "models")
VIDEO_PATH = os.path.join(os.environ["HOME"], "videos")

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--video', required=True, dest='video', type=str,
                    help="Name of video in video folder")
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


def process_video(video_path, batch_size=32, rate=0.5):
    split_name = os.path.splitext(os.path.basename(video_path))[0].split('-')
    timestamp = '-'.join(split_name[:-1])
    fps = int(split_name[-1])
    skip = int(rate * fps)
    initial = datetime.datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
        
    cap = cv2.VideoCapture(video_path)
    all_scores, all_classes, all_n, all_bboxes, timestamps  = [], [], [], [], []
    start_time = time.time()
    video_running = True
    processed = 0
    while video_running:
        frames = []
        for _ in range(skip):
            for _ in range(fps // 2):
                ret, frame = cap.read()
                if not ret:
                    print("Video finished")
                    video_running = False
                    break 
            if not video_running:
                break
            frames.append(frame)
            timestamps.append(str(initial + datetime.timedelta(seconds=rate*processed)))
            processed += 1
        if not frames:
            break
        bboxes, scores, n, classes = pred_from_frame(frames)
        all_scores.append(scores)
        all_bboxes.append(bboxes)
        all_n.append(n)
        all_classes.append(classes)
        if not video_running:
            break
    print('Frames processed: %d' % processed)
    print("Total time: {} seconds".format(int(time.time() - start_time)))
    full_bboxes = np.row_stack(all_bboxes)
    full_scores = np.row_stack(all_scores)
    full_classes = np.row_stack(all_classes)
    full_n = np.concatenate(all_n, axis=None)
    return timestamps, full_bboxes, full_scores, full_n, full_classes

def make_predictions(videoname):
    video = os.path.join(VIDEO_PATH, videoname)
    BATCH_SIZE = 72 
    start_time = time.time()
    timestamps, bboxes, scores, n, classes = process_video(video, batch_size=BATCH_SIZE)
    end_time = time.time()
    print('Elapsed: %f' % (end_time - start_time))
    
    # some cleaning
    agg_bboxes = []
    agg_scores = []
    num_frames = len(bboxes) 
    for ind in range(num_frames):
        image_n = int(n[ind])
        image_bboxes = bboxes[ind][:image_n]
        image_scores = scores[ind][:image_n]
        image_classes = classes[ind][:image_n]
        indices = image_classes == 1.
        image_bboxes = image_bboxes[indices]
        image_scores = image_scores[indices]
        agg_bboxes.append(image_bboxes.flatten().tolist())
        agg_scores.append(image_scores.tolist())

    # Save to Spark dataframe
    output_dir = os.path.join(OUTPUT_PATH, os.path.splitext(args.video)[0])
    spark = SparkSession.builder.getOrCreate()
    schema = StructType([StructField('timestamp', StringType(), True),
                         StructField('bboxes', ArrayType(DoubleType()), True),
                         StructField('scores', ArrayType(DoubleType()), True)])
    df = spark.createDataFrame(list(zip(*(timestamps, agg_bboxes, agg_scores))), schema)
    df.coalesce(1).write.mode('overwrite').json(output_dir)
    
    df1 = spark.read.json(output_dir)
    df1.show()
 
    # saving everything
#    filename = '{}-{}'.format(os.path.splitext(videoname)[0], CV_MODEL)
#    np.savez(filename, bboxes=bboxes, scores=scores, classes=classes,)
            
make_predictions(args.video)

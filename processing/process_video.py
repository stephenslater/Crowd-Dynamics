import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
import tqdm
import numpy as np
from pathlib import Path
import argparse
import cv2


sess = tf.Session()
graph = tf.get_default_graph()

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
    bboxes, score, classes = output['bboxes'], output['scores'], output['classes']
    return bboxes, score, classes


def process_video(video_path, batch_size=32, num_batches=100):
    cap = cv2.VideoCapture(video_path)
    all_scores, all_classes, all_bboxes = [], [], []
    start_time = time.time()
    video_running = True
    for _ in tqdm.tqdm(range(num_batches)):
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                print("Video finished")
                video_running = False
                break
            frames.append(frame)
        bboxes, scores, classes = pred_from_frame(frames)
        all_scores.append(scores)
        all_bboxes.append(bboxes)
        all_classes.append(classes)

        if not video_running:
            break
            
    print("Total time: {} seconds".format(int(time.time() - start_time)))
    full_bboxes = np.row_stack(all_bboxes)
    full_scores = np.row_stack(all_scores)
    full_classes = np.row_stack(all_classes)
    return full_bboxes, full_scores, full_classes


def make_predictions(videoname):
    video = os.path.join(VIDEO_PATH, videoname)
    BATCH_SIZE =16 
    NUM_BATCHES = 200
    start_time = time.time()
    bboxes, scores, classes = process_video(
        video, batch_size=BATCH_SIZE, num_batches=NUM_BATCHES)
    end_time = time.time()
    elapsed_time = np.array([end_time - start_time]) / BATCH_SIZE / NUM_BATCHES
    
    # saving everything
    filename = '{}-{}'.format(os.path.splitext(videoname)[0], CV_MODEL)
    np.savez(filename, bboxes=bboxes, scores=scores, classes=classes,
             inftime=elapsed_time)
            
make_predictions(args.video)

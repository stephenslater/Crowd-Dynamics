import base64
import json
from io import BytesIO

import numpy as np
import cv2
import requests
import time
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image



app = Flask(__name__)

DEFAULT_MODEL_NAME = 'object_detection'
TF_MODEL_SERVER = 'http://localhost:8501/v1/models'
CLIENTS_ROI_ALGOS = {}  # keeps a ROI algorithm for each client
MODEL_ALGORITHMS = {}  # Model selection algorithm for now.
MODELS = ['object_1', 'object_2']  # currently corresponds to ssd_mobilenet_v1 and resnet50
MODEL_TIMES = [0.15, 0.25]


def get_model_algorithm(client_id):
    """Uses model algoirthm corresponding with a specific client_id"""
    if client_id not in MODEL_ALGORITHMS:
        MODEL_ALGORITHMS[client_id] = NaiveModelSelection(MODELS, MODEL_TIMES)
    alg = MODEL_ALGORITHMS[client_id]
    return alg


def save_image_from_bytes(save_path, image_bytes, bboxes=None):
    """Saves image form base64 encoded bytes. Pass in bounding boxes in the
    form x1, y1, x2, y2 from 0-1 if you also want to draw prediction boxes"""
    image = cv2.imdecode(np.asarray(bytearray(image_bytes), dtype=np.uint8), 1)
    h, w, _ = image.shape
    if bboxes is not None:
        if len(bboxes) > 0:
            draw_boxes = np.array(np.array(bboxes) * np.array([w, h, w, h]), dtype=np.uint16)
            for x1, y1, x2, y2 in draw_boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    cv2.imwrite(save_path, image)


def rebuild_image_from_rois(images, rois, debug=False):
    """Rebuilds the image using ROI information. We assumet that the first image
    is the background, and followup images are the ones corresponding to the ROI
    images. Returns an encoded image in bytes corresonding to the stitched together
    image. Assume ROIs come in floats from 0 to 1"""
    if len(images) == 0:
        raise RuntimeError('Images were empty when trying to rebuild.')

    bg = cv2.imdecode(np.asarray(bytearray(base64.b64decode(images[0])), dtype=np.uint8), 1)
    h, w, _ = bg.shape
    # cv2.imwrite('bg.jpg', bg) 
    scaled_bboxes = np.array(np.array(rois) * np.array([w, h, w, h]), dtype=np.uint16)
    for im, roi in zip(images[1:], scaled_bboxes):
        x1, y1, x2, y2 = roi
        # print(roi)
        a = cv2.imdecode(np.asarray(bytearray(base64.b64decode(im)), dtype=np.uint8), 1)
        cv2.imwrite('a.jpg', a)
        # print(a.shape)
        # Overlay these images on the background
        bg[y1:y2, x1:x2] = a

    if debug:
        cv2.imwrite('test.jpg', bg)

    _, res = cv2.imencode('.jpg', bg)
    return res.tobytes()


def request_object_detection(image_path, model_name, target_size=(224, 224)):
    """Sends an image and requests a tf object detection prediction. Returns the
    raw output from the tf serving model. Assume that the network accepts a
    field `inputs` as uint8 serialized values. Also assumes that the network
    returns a field `predictions` as a list of predictions for images. For each,
    they shoudl have the structures `detection_scores`, `detection_boxes`, and
    `detection_classes`.
    @param image_path: path to image on disk
    @param model_name: name of tf serving model that is being requested

    @returns tuple (detection_classes, detection_boxes, detection_scores). Each
        are serialized as lists of the same length.
    """
    # Preprocessing our input image for most object detection
    # resizes the image and turns the pixels into values of uint8
    img = image.img_to_array(image.load_img(image_path, target_size=target_size))
    
    # BUG: this line might be added because of a bug in tf_serving(1.10.0-dev)
    # img = img.astype('float16')
    img = img.astype('uint8')

    payload = {"instances": [{'inputs': img.tolist()}]}

    # sending post request to TensorFlow Serving server
    start_inference_time = time.time()
    r = requests.post('%s/%s:predict' % (TF_MODEL_SERVER, model_name), json=payload)
    end_inference_time = time.time()
    inftime = end_inference_time - start_inference_time
    try:
        pred = json.loads(r.content.decode('utf-8'))
        ip = pred['predictions'][0]
        # Sometimes, we have a 'num_detections' that lets us filter number of boxes
        # IMPORTANT: TENSORFLOW SERVING MODEL PREDICTION returns as y1, x1, y2, x2
        # for SOME UNKNOWN REASON and as floats from 0-1
        # TODO: FIX THIS TOMORROW MORNING GRRRR
        if 'num_detections' in ip:
            a = int(ip['num_detections'])
            cls_ids = ip['detection_classes'][:a]
            scores = ip['detection_scores'][:a]
            bboxes = ip['detection_boxes'][:a]
            bboxes = [[x1, y1, x2, y2] for y1, x1, y2, x2 in bboxes]
            return cls_ids, scores, bboxes, inftime

        cls_ids = ip['detection_classes']
        scores = ip['detection_scores']
        bboxes = ip['detection_boxes']
        bboxes = [[x1, y1, x2, y2] for y1, x1, y2, x2 in bboxes]
        return cls_ids, scores, bboxes, inftime 
    except Exception as err:
        print(str(err))
        print('Contents of response: %s' % str(r.content))
        raise RuntimeError('Tensorflow model response failed...')


def generate_roi(client_id, bboxes):
    """Handles generating ROI for a specific client and prediction boxes"""
    if client_id not in CLIENTS_ROI_ALGOS:
        CLIENTS_ROI_ALGOS[client_id] = SimpleROI()
    roi_alg = CLIENTS_ROI_ALGOS[client_id]
    roi_alg.get_prediction(bboxes)
    return roi_alg.roi()



@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/testing/', methods=['GET', 'POST'])
def test_no_inference():
    inftime_start = time.time()
    if 'client_id' not in request.json:
        return 'You need to specify a client_id field in your request'
    cid = request.json['client_id']
    model_alg = get_model_algorithm(cid)
    if 'quality' in request.json:
        model_alg.record_quality(int(request.json['quality']))
    if 'rtt' in request.json:
        model_alg.record_latency(float(request.json['rtt']))
    bound = None if 'bound' not in request.json else float(request.json['bound'])
    model = model_alg.model(bound)

    if 'model' in request.json and request.json['model'] in MODELS:
        model = request.json['model']
        print('Using hardcoded model: %s' % model)
    else:
        print('Using optimal model: %s' % model)

    if 'latency' in request.json:
        time.sleep(float(request.json['latency']))

    # If the user commanded the server save the predictions and filename, then
    # the server can do so for debugging and visualization purposes.
    if 'save' in request.json:
        # Normal image used for inference, we just do the same thing
        byte_image = base64.b64decode(request.json['b64'])
        save_image_from_bytes(request.json['save'], byte_image)

    inftime = time.time() - inftime_start
    # HANDLING RESPONSE TO CLIENT
    resp = dict(predtime=inftime)

    # Returning JSON response to the frontend
    # Turn how large the other person's packet was back for record-keeping purposes
    if request.content_length:
        resp['request_size'] = int(request.content_length)
    # Reattach the timestamps if the sender attached
    if 'timestamp' in request.json:
        resp['timestamp'] = request.json['timestamp']
    return jsonify(resp)


@app.route('/object/predict/', methods=['POST'])
def classify():
    if 'client_id' not in request.json:
        return 'You need to specify a client_id field in your request'
    cid = request.json['client_id']
    model_alg = get_model_algorithm(cid)
    if 'quality' in request.json:
        model_alg.record_quality(int(request.json['quality']))
    if 'rtt' in request.json:
        model_alg.record_latency(float(request.json['rtt']))
    bound = None if 'bound' not in request.json else float(request.json['bound'])
    model = model_alg.model(bound)

    if 'model' in request.json and request.json['model'] in MODELS:
        model = request.json['model']
        print('Using hardcoded model: %s' % model)
    else:
        print('Using optimal model: %s' % model)

    if 'latency' in request.json:
        time.sleep(float(request.json['latency']))

    if 'b64' in request.json:
        try:
            # Decoding and pre-processing base64 image
            cls_ids, scores, bboxes, inftime = request_object_detection(
                BytesIO(base64.b64decode(request.json['b64'])), model)
        except Exception as err:
            return str(err)

    # If the user commanded the server save the predictions and filename, then
    # the server can do so for debugging and visualization purposes.
    if 'save' in request.json:
        # Normal image used for inference, we just do the same thing
        byte_image = base64.b64decode(request.json['b64'])
        save_image_from_bytes(request.json['save'], byte_image, bboxes=bboxes)

    # HANDLING RESPONSE TO CLIENT
    resp = dict(cls_ids=cls_ids, scores=scores, bboxes=bboxes, predtime=inftime)

    # Returning JSON response to the frontend
    # Turn how large the other person's packet was back for record-keeping purposes
    if request.content_length:
        resp['request_size'] = int(request.content_length)
    # Reattach the timestamps if the sender attached
    if 'timestamp' in request.json:
        resp['timestamp'] = request.json['timestamp']
    return jsonify(resp)


@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Handle the case with regular images
    if 'b64' in request.json:
        try:
            # Decoding and pre-processing base64 image
            cls_ids, scores, bboxes, inftime = request_object_detection(
                BytesIO(base64.b64decode(request.json['b64'])), DEFAULT_MODEL_NAME)
        except Exception as err:
            return str(err)
    elif 'rois' in request.json:
        print('Found ROIs')
        a = request.json
        image_bytes = list(a['images'])
        rois = list(a['rois'])
        rebuilt_im = rebuild_image_from_rois(image_bytes, rois)
        try:
            # Decoding and pre-processing base64 image
            cls_ids, scores, bboxes, inftime = request_object_detection(
                BytesIO(rebuilt_im), DEFAULT_MODEL_NAME)
        except Exception as err:
            return jsonify({'error': str(err)})

    # If the user commanded the server save the predictions and filename, then
    # the server can do so for debugging and visualization purposes.
    if 'save' in request.json:
        if 'rois' in request.json:
            image = cv2.imdecode(np.asarray(bytearray(rebuilt_im), dtype=np.uint8), 1)
            # bboxes are floats from 0-1, need to scale back to image size.
            h, w, _ = image.shape
            if len(bboxes) > 0:
                draw_boxes = np.array(np.array(bboxes) * np.array([w, h, w, h]), dtype=np.uint16)
                for x1, y1, x2, y2 in draw_boxes:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
            cv2.imwrite(request.json['save'], image)
        else:
            # Normal image used for inference, we just do the same thing
            byte_image = base64.b64decode(request.json['b64'])
            image = cv2.imdecode(np.asarray(bytearray(byte_image), dtype=np.uint8), 1)
            h, w, _ = image.shape
            if len(bboxes) > 0:
                draw_boxes = np.array(np.array(bboxes) * np.array([w, h, w, h]), dtype=np.uint16)
                for x1, y1, x2, y2 in draw_boxes:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            cv2.imwrite(request.json['save'], image)

    # HANDLING RESPONSE TO CLIENT
    resp = dict(cls_ids=cls_ids, scores=scores, bboxes=bboxes, predtime=inftime)
    # Handle generating ROI if specified (need client_id to generate)
    if 'client_id' in request.json:
        clid = request.json['client_id']
        rois = generate_roi(clid, bboxes)
        resp['rois'] = rois

    # Returning JSON response to the frontend
    # Turn how large the other person's packet was back for record-keeping purposes
    if request.content_length:
        resp['request_size'] = int(request.content_length)
    # Reattach the timestamps if the sender attached
    if 'timestamp' in request.json:
        resp['timestamp'] = request.json['timestamp']
    return jsonify(resp)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5900, type=int)
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)

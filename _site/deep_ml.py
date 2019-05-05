from pyspark import SparkConf
from pyspark import SparkContext

conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[1]')
        .set('spark.executor.memory', '45G')
        .set('spark.driver.memory', '45G')
        .set('spark.driver.maxResultSize', '10G'))
sc = SparkContext(conf=conf)


from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.image import ImageSchema

# Might want to use this command
# pyspark --packages databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11
# https://github.com/databricks/spark-deep-learning#working-with-images-in-spark

from sparkdl import DeepImageFeaturizer
from sparkdl import DeepImagePredictor
from sparkdl import TFImageTransformer
import sparkdl.graph.utils as tfx
from sparkdl.transformers import utils
import tensorflow as tf
import os

CV_MODEL = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_PATH = os.path.join(os.environ['HOME'], "models", CV_MODEL)
print(os.listdir(MODEL_PATH))

frozen_file = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
with tf.gfile.GFile(frozen_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

input_nodes = [v for v in graph.as_graph_def().node if "Placeholder" in v.op]
output_nodes = [v for v in graph.as_graph_def().node if "detection" in v.name]
print(input_nodes)
print(output_nodes)



# Make the new graph with the image placeholder for spark
with tf.Session(graph=graph) as sess:
    image_arr = utils.imageInputPlaceholder()
    resized_images = tf.image.resize_images(image_arr, (299, 299))
    frozen_graph = tfx.strip_and_freeze_until([resized_images],
            graph, sess, return_graph=True)

transformer = TFImageTransformer(inputCol='image', outputCol="predictions",
        graph=graph, inputTensor=image_arr,
        outputTensor=resized_images,
        outputMode="image", channelOrder="RGB")

HOME = os.environ['HOME']
image_dir = os.path.join(HOME, "videos/chicken/000000.png")
image_df = ImageSchema.readImages(image_dir)
print(image_df.count())

predictions_df = transformer.transform(image_df)

predictions_df.take(1)




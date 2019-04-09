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

image_dir = "/Users/wfu/videos/chicken"
image_df = ImageSchema.readImages(image_dir)

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels", modelName="InceptionV3", decodePredictions=True, topK=10)
predictions_df = predictor.transform(image_df)

print(predictions_df.take(1))




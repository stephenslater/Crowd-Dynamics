#!/bin/sh

spark-submit \
--packages databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11 \
deep_ml.py

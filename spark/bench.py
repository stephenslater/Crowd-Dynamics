"""Benchmarking and evaluating the speedup for our historical analytics. We
scan through up to 4 cores and 8 executors, since we used an EMR cluster with
8 m4.xlarge workers (and each m4.xlarge worker has 4 vCPUs). However, this
could be changed depending on the infrastructure you run it on."""
import sys
import os
import subprocess
import time
import pickle
from subprocess import DEVNULL, STDOUT

SPARK = "spark-submit"
FILE = "bench_spark.py"

cores2runtime = {}

command = "{} --master local {}".format(SPARK, FILE)
start_time = time.time()
subprocess.run(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)
time_elapsed = time.time() - start_time
print('serial')
print(time_elapsed)
cores2runtime['serial'] = time_elapsed

for cores in range(1, 5):
    for num_executors in range(1, 9):
        command = "{} --num-executors {} --executor-cores {} {}".format(SPARK, num_executors, cores, FILE)
        start_time = time.time()
        subprocess.run(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        time_elapsed = time.time() - start_time
        print("num_executors", num_executors)
        print("cores", cores)
        print(time_elapsed)
        cores2runtime[(num_executors, cores)] = time_elapsed

print(cores2runtime)

# Write the results to a file to load later.
with open("spark_analytics_benchmark.pkl", "wb") as f:
    pickle.dump(cores2runtime, f)
    
"""Bench Pressing with Spark"""
import sys
import os
import subprocess
import time
import pickle
from subprocess import DEVNULL, STDOUT


SPARK = "spark-submit --deploy-mode cluster"
FILE = "wfus.py"

cores2runtime = {}

for cores in [4, 3, 2, 1]:
    for num_executors in [3, 2, 1]:
        command = "{} --num-executors {} --executor-cores {} {}".format(
            SPARK, num_executors, cores, FILE)
        start_time = time.time()
        subprocess.run(command, shell=True, stdout=DEVNULL)
        time_elapsed = time.time() - start_time
        print("num_executors", num_executors)
        print("cores", cores)
        print(time_elapsed)
        cores2runtime[(num_executors, cores)] = time_elapsed

print(cores2runtime)

with open("testdayscan.pkl", "wb") as f:
    pickle.dump(cores2runtime, f)
    

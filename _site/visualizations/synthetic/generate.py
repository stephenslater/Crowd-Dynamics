"""Generating synthetic data to test with before getting the real results
from Spark and Databricks. Doing this while everything is running and to see
if the visualizations look good."""
import numpy as np
import pandas as pd
import pathlib
import datetime

# Let's generate 7 days of data, named day{i}.csv
for i in range(7):
    folder = pathlib.Path.cwd() / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    path = pathlib.Path.cwd() / 'data' / 'day{}.csv'.format(i)
    fake_date = datetime.date.today() - datetime.timedelta(days=i)
    
    recs = []
    # generate some data from all hours
    for hour in range(24):
        fake_time = datetime.datetime.combine(fake_date, datetime.time(hour))
        time_string = fake_time.isoformat()
        velocity = float(np.random.uniform(0, 10))
        recs.append([time_string, velocity])
    
        df = pd.DataFrame(recs)
        df.columns = ["timestamp", "velocity"]
    
    df.to_csv(path)

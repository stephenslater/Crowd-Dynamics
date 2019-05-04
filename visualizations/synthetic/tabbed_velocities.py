import pandas as pd
import numpy as np
from pathlib import Path

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import Tabs, Panel

# We are planning for each day of analytics to be a .CSV
# I will fake some data.

figs = {} 
for i in range(7):
    datapath = Path.cwd() / 'data' / "day{}.csv".format(i)
    df = pd.read_csv(datapath)
    df.timestamp = df.timestamp.apply(str)
    df.index.name = 'index'
    hover = HoverTool()
    hover.tooltips = [('Timestamp', '@timestamp')]
    p = figure(title = "Average Velocities over Day",
               plot_height=500, 
               plot_width=500,
               tools=[hover, "pan,reset,wheel_zoom"])
    # print(df.info())
    p.vbar(x='index', 
           top='velocity',
           width=0.9,
           color='red',
           source=df)
    p.xaxis.axis_label = "Hour in Day (24 hour format)"
    p.yaxis.axis_label = "Average Velocity (m/s)"
    figs[i] = p

# Create panels for each day for visualization.
panels = [Panel(child=p, title="Day {}".format(i))
          for i, p in figs.items()]
tabs = Tabs(tabs=panels)
show(tabs)

"""
riser_collapse.py

Simulate collapse of terrace risers.
"""

# Import numpy, Landlab components, etc. for running the model
import numpy as np
import pandas as pd
from landlab import RasterModelGrid
from landlab.io import read_esri_ascii
from landlab.components import LinearDiffuser
from landlab.components import FlowRouter
from landlab.components import StreamPowerEroder
from landlab.components import PrecipitationDistribution
from landlab.components import FlowDirectorD8

# Import components for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, plot, xlabel, ylabel, title, subplot
import numpy as np
from landlab.plot import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

# Add random numbers
import random
from numpy.random import RandomState

# Add deepcopy
import copy
from copy import deepcopy

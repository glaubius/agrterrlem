""""
LEM_SensitivityAnalysis

Python file for running sensitivity analysis for LEM with risers and riser
collapse.
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


## INITIALIZE

# Set random seed so figures are reproducible
np.random.seed(1337)

# Create artificial area, 50 x 50, 1.0 m cell resolution
rmg = RasterModelGrid((50, 50), 1.)
rmg.axis_units = ('m', 'm')
# Add slope and random noise to topography
init_topo = ((200 - rmg.node_x) * 0.03) + 100. + np.random.normal(0, 0.005, 2500)

# Add node IDs as field to RasterModelGrid for help changing node values later
node_ID = rmg.nodes
rmg.add_field('node', 'node_ID', node_ID, noclobber=True)

# Create field for terrace wall locations
terrWallLoc = np.zeros(2500)

# Add terraces every 5 meters
for i in range(5, 90, 5):
    terr_loc = np.where((rmg.node_x) <= i)
    init_topo[terr_loc] += 2.0
    # add terrace wall location to field
    terrwallloc = np.where((rmg.node_x) == i)
    terrWallLoc[terrwallloc] += 1.

# Add initial topography to raster model grid for later plotting
rmg.add_field('node', 'initial_topographic__elevation', init_topo, noclobber=True)

# Create separate elevation data so initial topography is not overwritten
z = copy.deepcopy(init_topo)

# Add elevation for topography to raster model grid
rmg.add_field('node', 'topographic__elevation', z, noclobber=False)

# Add terrace wall location field to rmg
rmg.add_field('node', 'terrace_wall__location', terrWallLoc)

# Set the boundary conditions: all NODATA nodes in the DEM are closed
# boundaries, and the outlet is set to an open boundary.
rmg.set_watershed_boundary_condition(z)

# Create list of wall nodes
walls = list(np.where(rmg.at_node['terrace_wall__location'] == 1.))
lenWalls = (len(rmg.at_node['terrace_wall__location'][walls]))

# Set maximum values for x and y coordinates (needed to find downhill node)
(max_x, max_y) = rmg.shape
max_x = float(max_x)
max_y = float(max_y)

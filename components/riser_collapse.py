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


## Determine downhill node from given node
def determine_downhill_node(ID, max_x, max_y):

    node_elev = rmg.at_node['topographic__elevation'][ID]
    nodeX = rmg.node_x[ID]
    nodeY = rmg.node_y[ID]

    coords_by_neighbor = [
        [nodeX - 1, nodeY - 1],
        [nodeX - 1, nodeY],
        [nodeX - 1, nodeY + 1],
        [nodeX, nodeY + 1],
        [nodeX + 1, nodeY + 1],
        [nodeX + 1, nodeY],
        [nodeX + 1, nodeY - 1],
        [nodeX, nodeY - 1]
    ]

    # Initialize list
    neighbors = []

    for v in range(0, 8):
        coords = 0
        coords = coords_by_neighbor[v]
        v_x = int((coords)[0])
        v_y = int((coords)[1])
        if (v_x < 0) or (v_x >= max_x):
            pass
        elif (v_y < 0) or (v_y >= max_y):
            pass
        else:
            v_ID = rmg.grid_coords_to_node_id(v_y, v_x)
            v_elev = rmg.at_node['topographic__elevation'][v_ID]
            elev_diff = node_elev - v_elev
            neighbors.append([v_ID, elev_diff])


    # Determine ID of downhill node
    neighbors.sort(key=lambda cell: cell[1], reverse=True)
    downhill = neighbors[0]
    down_ID = downhill[0]
    return down_ID

    neighbors.clear()

def riser_collapse():
    for i in range(lenWalls):
        ID = rmg.at_node['node_ID'][walls][i]
        prob = np.random.rand(1)
        if prob < 0.5:
            collapse_size = np.random.normal(2, 0.5, 1)
            print("Collapse size is {collapse_size} m.".format(collapse_size=collapse_size))
            print("Testing node {node_ID} with elev {elev}".format(node_ID=rmg.at_node['node_ID'][walls][i], elev=rmg.at_node['topographic__elevation'][walls][i]))
            print("Initial elevation of node {node_ID} is {elev}".format(node_ID=ID, elev=z[ID]))
            downhill_node = determine_downhill_node(ID)
            print("Initial elevation of downhill node {downhill} is {elev}".format(downhill=downhill_node,elev=z[downhill_node]))
            if (z[ID] < z[downhill_node]):
                print("impossible!")
            else:
                z[ID] -= collapse_size
                print("New elevation of node is {elev}".format(elev=z[ID]))
                z[downhill_node] += collapse_size
                print("New elevation of downhill node is {}".format(z[downhill_node]))
                terrWallLoc[ID] = 0.
                print("Terrace wall location value for collapsed node is {node}".format(node=rmg.at_node['terrace_wall__location'][walls][i]))
        else:
            pass

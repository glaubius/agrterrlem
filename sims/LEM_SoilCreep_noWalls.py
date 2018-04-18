"""
LEM_SoilCreep_noWalls

Simulation of soil creep in terraced landscape without modifications for
terrace risers for FIG in Methods section.
"""

### FUNCTION FOR PLOTTING INITIAL AND FINAL PROFILES
def plot_1D_2lines(rmg,
            surface1,
            surface2,
            profile_start,
            profile_end,
            title='Topographic Profile'):
    """ Generate a 1D profile plot of topography or other surface.

    Parameters
    ----------
    rmg : RasterModelGrid
        RasterModelGrid created using Landlab containing topographic__elevation
        or other surface to be plotted.
    surface1 : field name
        Surface field name. For 1st line on plot (initial).
    surface2 : field name
        Surface field name. For 2nd line on plot (final).
    profile_start : tuple of 2 integers
        Coordinates for starting (southeastern) point of profile. Must be a tuple
        of 2 integers, such as (1, 2).
    profile_end : tuple of 2 integers
        Coordinates for ending (northwestern) point of profile. Must be a tuple
        of 2 integers, such as (8, 9).
    title : string
        Title for plot. Defaults to 'Topographic profile' unless supplied.

    """
    plt.close() ## close plot to prevent multiple colorbars

    fig = plt.figure(title, figsize=(8.0, 5.0))

    profile_start = list((profile_start))
    profile_end = list((profile_end))

    ## Set start and end of each profile in x and y
    grid_shape = list((rmg.shape))
    x_len = grid_shape[1]
    y_len = grid_shape[0]

    start_x = profile_start[0]
    assert (start_x <= x_len), "Given coordinate is not within the grid"

    start_y = profile_start[1]
    assert (start_y <= y_len), "Given coordinate is not within the grid"

    end_x = profile_end[0]
    assert (end_x <= x_len), "Given coordinate is not within the grid"

    end_y = profile_end[1]
    assert (end_y <= y_len), "Given coordinate is not within the grid"

    ## Calculate differences between start and end x and y
    diff_x = abs(end_x - start_x)
    diff_y = abs(end_y - start_y)

    ## Determine if x or y will be horizontal axis for profile
    if diff_y > diff_x:
        # create y_list
        y_list = list(range(start_y, end_y + 1, 1))
        ## create z_list
        # reshape surface
        Z1 = rmg.at_node[surface1].reshape(rmg.shape)
        Z2 = rmg.at_node[surface2].reshape(rmg.shape)
        # create list for non-axis (x_list)
        x_list = np.linspace(start_x, end_x, len(y_list))
        for x in range(0, len(x_list)):
            x_list[x] = round(x_list[x])
        # use y_list and x_list to populate z_list
        z1_list = []
        z2_list = []
        for i in range(0, len(y_list)):
            z1_list.append(Z1[x_list[i], y_list[i]])
            z2_list.append(Z2[x_list[i], y_list[i]])
        # plot(y_list, z_list)
        fig = plt.figure()
        plt.xlabel('Northing')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(y_list, z1_list, c='0.75', ls='-', lw=4, label='Initial topography')
        plt.plot(y_list, z2_list, 'r--', label='Final topopgraphy')
        plt.legend(loc='best')
    else:
        # create x_list
        x_list = list(range(start_x, end_x + 1, 1))
        ## create z_list
        # reshape surface
        Z1 = rmg.at_node[surface1].reshape(rmg.shape)
        Z2 = rmg.at_node[surface2].reshape(rmg.shape)
        # create list for non-axis (y_list)
        y_list = np.linspace(start_y, end_y, len(x_list))
        for y in range(0, len(y_list)):
            y_list[y] = round(y_list[y])
        # use x_list and y_list to populate z_list
        z1_list = []
        z2_list = []
        for i in range(0, len(x_list)):
            z1_list.append(Z1[x_list[i], y_list[i]])
            z2_list.append(Z2[x_list[i], y_list[i]])
        # plot(x_list, z_list)
        fig = plt.figure()
        plt.xlabel('Easting')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(y_list, z1_list, c='0.75', ls='-',  lw=4, label='Initial topography')
        plt.plot(y_list, z2_list, 'r--', label='Final topopgraphy')
        plt.legend(loc='best')
    plt.show()

def save_1D_2lines(rmg,
            surface1,
            surface2,
            profile_start,
            profile_end,
            title='Topographic Profile',
            output_location='.',
            tag='profile_plot',
            fileType='pdf'):
    """ Generate a 1D profile plot of topography or other surface.

    Parameters
    ----------
    rmg : RasterModelGrid
        RasterModelGrid created using Landlab containing topographic__elevation
        or other surface to be plotted.
    surface1 : field name
        Surface field name. For 1st line on plot (initial).
    surface2 : field name
        Surface field name. For 2nd line on plot (final).
    profile_start : tuple of 2 integers
        Coordinates for starting (southeastern) point of profile. Must be a tuple
        of 2 integers, such as (1, 2).
    profile_end : tuple of 2 integers
        Coordinates for ending (northwestern) point of profile. Must be a tuple
        of 2 integers, such as (8, 9).
    title : string
        Title for plot. Defaults to 'Topographic profile' unless supplied.
    output_location : string
        Location where plot will be stored.
    tag : string
        Name for plot file, include information about type of plot, year, etc.
    fileType : string
        Type of file for plot. Can be pdf, png, jpg, etc.
    """
    plt.close() ## close plot to prevent multiple colorbars

    fig = plt.figure(title, figsize=(8.0, 5.0))

    profile_start = list((profile_start))
    profile_end = list((profile_end))
    start_x = profile_start[0]
    start_y = profile_start[1]
    end_x = profile_end[0]
    end_y = profile_end[1]

    ## Calculate differences between start and end x and y
    diff_x = end_x - start_x
    diff_y = end_y - start_y

    ## Determine if x or y will be horizontal axis for profile
    if diff_y > diff_x:
        # create y_list
        y_list = list(range(start_y, end_y + 1, 1))
        ## create z_list
        # reshape surface
        Z1 = rmg.at_node[surface1].reshape(rmg.shape)
        Z2 = rmg.at_node[surface2].reshape(rmg.shape)
        # create list for non-axis (x_list)
        x_list = np.linspace(start_x, end_x, len(y_list))
        for x in range(0, len(x_list)):
            x_list[x] = round(x_list[x])
        # use y_list and x_list to populate z_list
        z1_list = []
        z2_list = []
        for i in range(0, len(y_list)):
            z1_list.append(Z1[x_list[i], y_list[i]])
            z2_list.append(Z2[x_list[i], y_list[i]])
        # plot(y_list, z_list)
        fig = plt.figure()
        plt.xlabel('Northing')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(y_list, z1_list, c='0.75', ls='-', lw=4, label='Initial topography')
        plt.plot(y_list, z2_list, 'r--', label='Final topopgraphy')
        plt.legend(loc='best')
    else:
        # create x_list
        x_list = list(range(start_x, end_x + 1, 1))
        ## create z_list
        # reshape surface
        Z1 = rmg.at_node[surface1].reshape(rmg.shape)
        Z2 = rmg.at_node[surface2].reshape(rmg.shape)
        # create list for non-axis (y_list)
        y_list = np.linspace(start_y, end_y, len(x_list))
        for y in range(0, len(y_list)):
            y_list[y] = round(y_list[y])
        # use x_list and y_list to populate z_list
        z1_list = []
        z2_list = []
        for i in range(0, len(x_list)):
            z1_list.append(Z1[x_list[i], y_list[i]])
            z2_list.append(Z2[x_list[i], y_list[i]])
        # plot(x_list, z_list)
        fig = plt.figure()
        plt.xlabel('Easting')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(y_list, z1_list, c='0.75', ls='-',  lw=4, label='Initial topography')
        plt.plot(y_list, z2_list, 'r--', label='Final topopgraphy')
        plt.legend(loc='best')
    plt.savefig("{}/{}.{}".format(output_location, tag, fileType), bbox_inches='tight')
    plt.close()

# Import numpy, Landlab components, etc. for running the model
import numpy as np
from landlab import RasterModelGrid
from landlab.io import read_esri_ascii
from landlab.components import LinearDiffuser
from landlab.components import FlowRouter
from landlab.components import StreamPowerEroder

# Import components for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, plot, xlabel, ylabel, title, subplot
import numpy as np
from landlab.plot import imshow_grid

# Add random numbers
import random
from numpy.random import RandomState

# Add deepcopy
import copy
from copy import deepcopy

### INITIALIZE
# Set random seed so figures are reproducible
np.random.seed(1337)

# Create artificial area, 50 x 50, 1.0 m cell resolution
rmg = RasterModelGrid((50, 50), 1.)
rmg.axis_units = ('m', 'm')
# Add slope and random noise to topography
init_topo = ((200 - rmg.node_x) * 0.03) + 100. + np.random.normal(0, 0.005, 2500)
# Add terraces every 5 meters
for i in range(5, 90, 5):
    terr_loc = np.where((rmg.node_x) <= i)
    init_topo[terr_loc] += 2.0

# Add initial topography to raster model grid for later plotting
rmg.add_field('node', 'initial_topographic__elevation', init_topo, noclobber=True)

# Create separate elevation data so initial topography is not overwritten
z = copy.deepcopy(init_topo)

# Add elevation for topography to raster model grid
rmg.add_field('node', 'topographic__elevation', z, noclobber=False)

# Set the boundary conditions: all NODATA nodes in the DEM are closed
# boundaries, and the outlet is set to an open boundary.
rmg.set_watershed_boundary_condition(z)

# Set landscape parameters
uplift_rate = 0.0001 # [m/yr]
K_sp = 1.E-4 # units vary depending on m_sp and n_sp
m_sp = 0.5
n_sp = 1.
K_hs = 0.03 # [m^2/yr] for diffusivity; 0.2

### Initialize components
# Linear Diffuser component for soil creep
ld = LinearDiffuser(rmg, linear_diffusivity=K_hs)

def run_LEM(years):

    for i in range(years):
        # Soil creep
        ld.run_one_step(1.)
        # Uplift
        z[rmg.core_nodes] += uplift_rate  # add the uplift m/y

        # Add some output to let us see the model isn't hanging and output
        # profile plot every 5 steps/years to show changing landscape:
        if i % 5 == 0:
            print(i)
            #save_1D_2lines(rmg, 'initial_topographic__elevation', 'topographic__elevation', (2, 2), (9, 9), "Initial and Final Topography - Year %d" % (i), '.', "profile_plot_year%d" % (i))


    # Calculate and output mean lowering rate (m yr-1)
    topoDiff = init_topo - z
    sumTopoDiff = sum(topoDiff)
    aveTopoDiff = sumTopoDiff / 2500
    meanElevLowering = aveTopoDiff / years
    #print(sumTopoDiff)
    print("Average difference in elevation between initial and final topography is %f m." % (aveTopoDiff))
    print("Mean Elevation Lowering over %d years is %f meters per year." % (years, meanElevLowering))

    # Plot profile of entire landscape initial and Final
    #save_1D_2lines(rmg, 'initial_topographic__elevation', 'topographic__elevation', (2, 2), (45, 45), "Initial and Final Topography - Year %d" % (i), '.', "profile_plot_yearEntire%d" % (i))


    # Plot profile of detail of landscape initial and Final
    #save_1D_2lines(rmg, 'initial_topographic__elevation', 'topographic__elevation', (2, 2), (9, 9), "Initial and Final Topography - Year %d" % (i), '.', "profile_plot_year%d" % (i))

run_LEM(100)

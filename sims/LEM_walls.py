"""
LEM_walls.py

Implementation of simple LEM in Landlab that includes walls to
demonstrate how terraced terrain morphology should retain terrace morphology during
LEM simulations. Modeled landscape is artificially created to represent Cinque
Terre (specificially Vernazza). Slope, terrace dimensions, rainfall, and erosion
rates are similar to those in Cinque Terre (see below).
For Glaubius et al., in prep.

Cinque Terre Data for Modeling:

    LANDSCAPE: Tread width 3-4 meters (TerranovaEtAl2006, p. 119),
    riser height in Vernazza 1.5 - 2.5 m (GalveEtAl2015, p. 103),
    riser width varies between 50 and 100 cm based on height (Manual pp. 23-24),
    Average wall cross section is 1.25 m2 (ALPTER)
        50 x 50 square landscape, 1.0 m resolution, 4 m tread, 2 m wall height,
        1 m wall width (due to cell size)

    RAINFALL: 902 mm per year (climate data)

    EROSION/LOWERING RATES: Rates for similar Italian watersheds (although not
    known if those watersheds are terraced) from RompaeyEtAl2005, Table 1.
    Range of observed SSY (t ha-1 yr-1): 0.1 to 19.6; mean is 4.55.
    Coverted SSY to lowering rate (m yr-1): 0.000006 to 0.001245

Written by Jennifer Glaubius, March 2018

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
from landlab.plot import imshow_grid

# Add random numbers
import random
from numpy.random import RandomState

# Add deepcopy
import copy
from copy import deepcopy

### INITIALIZE LEM

# Set random seed so figures are reproducible
np.random.seed(1337)

# Create artificial area, 50 x 50, 1.0 m cell resolution
rmg = RasterModelGrid((50, 50), 1.)
rmg.axis_units = ('m', 'm')
# Add slope and random noise to topography
init_topo = ((200 - rmg.node_x) * 0.03) + 100. + np.random.normal(0, 0.005, 2500)
# Add terraces every 5 meters
for i in range(5, 90, 5):
    terr_loc = np.where((rmg.node_x + rmg.node_y) <= i)
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

# Set spatially variable diffusivity for soil creep
df = pd.Series(data=np.array(rmg.at_node['terrace_wall__location']), name="walls")
df1 = df.to_frame(name="walls")
df1['Kd'] = '0'
df1.loc[df1.walls == 1., 'Kd'] = 0.
df1.loc[df1.walls == 0., 'Kd'] = 1.0
dfList = df1['Kd'].tolist()
Kd = np.array(dfList)

# Add Kd to rmg
rmg.add_field('node', 'Kd', Kd)

# Set links to kd value for uphill nodes
maxField= map_value_at_max_node_to_link(rmg, 'topographic__elevation', 'Kd')
rmg.add_field('link', 'Kd_maxField', maxField)


# Set spatially variable diffusivity for erosivity (Ksp)
df = pd.Series(data=np.array(rmg.at_node['terrace_wall__location']), name="walls")
df1 = df.to_frame(name="walls")
df1['Ksp'] = '0'
df1.loc[df1.walls == 1., 'Ksp'] = 0.
df1.loc[df1.walls == 0., 'Ksp'] = 0.00001
dfList = df1['Ksp'].tolist()
Ksp = np.array(dfList)

# Set initial surface_water__discharge
rmg['node']['surface_water__discharge'] = np.zeros(rmg.number_of_nodes)
Q = rmg.at_node['surface_water__discharge']

### Initialize components
# Linear Diffuser component for soil creep
ld = LinearDiffuser(rmg, linear_diffusivity='Kd_maxField')

# Flow router using D8 Flow Director (default FD)
fr = FlowRouter(rmg)

# Stream Power Eroder (erosion by water)
sp = StreamPowerEroder(rmg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp, threshold_sp=0, use_Q=Q)


### SET UP SIMULATION
# Define function to run model based on
# number of years provided
# Discharge based on rainfall from lines 154-158 of
# https://github.com/landlab/landlab/blob/master/landlab/components/stream_power/stream_power.py
def run_LEM(years):
    # Create list for saving rainfall
    rainOutput = []
    for i in range(years):
        ### Random rainfall
        # Pull random number for yearly rain total based on mean and sd of RAINFALL
        # in study area
        yearRain = np.random.normal(0.878982,0.2744,1)
        # Add rainfall to rainfall list
        rainOutput.append(yearRain)
        # Add randomness to yearRain (annual rain total) over the landscape
        rain = yearRain + 0.1*np.random.rand(rmg.number_of_nodes)
        # Add rain to water__unit_flux_in for flow routing
        _ = rmg.add_field('water__unit_flux_in',
                          rain,
                          at = 'node',
                         noclobber=False)

        # Route flow
        fr.run_one_step()
        Q = rmg.at_node['surface_water__discharge']
        # Erosion by water (stream power)
        sp.run_one_step(1., use_Q=Q)
        # Soil creep
        ld.run_one_step(1.)
        # Uplift
        z[rmg.core_nodes] += uplift_rate  # add the uplift m/y

        # Add some output to let us see the model isn't hanging and output
        # profile plot every 5 steps/years to show changing landscape:
        if i % 5 == 0:
            print(i)
            ## Turn on the line below to save plots to file
            #save_1D_2lines(rmg, 'initial_topographic__elevation', 'topographic__elevation', (2, 2), (9, 9), "Initial and Final Topography - Year %d" % (i), '.', "profile_plot_year%d" % (i))


    # Calculate and output mean annual rainfall (m)
    sumRain = sum(rainOutput)
    aveRain = sumRain / years
    print("Average rainfall is %f m per year." % (aveRain))

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


### SIMULATE
run_LEM(100)

"""
Generate 1D profile plots for modeling results.

This component outputs a 1D profile plot using given start and end points. The
component is partially based on instructions for creating 1D plots in Landlab
in this tutorial: https://nbviewer.jupyter.org/github/landlab/tutorials/blob/master/plotting/landlab-plotting.ipynb
Rather than taking east-west slices at certain intervals along the y-axis, however,
this component creates the plot from user-supplied start and end points.

..codeauthor:: Jennifer Glaubius

Written August 2017

Examples
--------

"""

import matplotlib.pylab as plt
from matplotlib.pyplot import title, show, figure, plot, subplot, xlabel, ylabel
import numpy as np

def plot_1D(rmg,
            surface,
            profile_start,
            profile_end,
            title='Topographic Profile'):
    """ Generate a 1D profile plot of topography or other surface.

    Parameters
    ----------
    rmg : RasterModelGrid
        RasterModelGrid created using Landlab containing topographic__elevation
        or other surface to be plotted.
    surface : field name
        Surface field name. Defaults to 'topographic__elevation' unless supplied.
    profile_start : tuple of 2 integers
        Coordinates for starting (southeastern) point of profile. Must be a tuple
        of 2 integers, such as (1, 2) in (x, y) ordering.
    profile_end : tuple of 2 integers
        Coordinates for ending (northwestern) point of profile. Must be a tuple
        of 2 integers, such as (8, 9) in (x, y) ordering.
    title : string
        Title for plot. Defaults to 'Topographic profile' unless supplied.
    output : string
        Set output to either be plotted on screen or saved to file. Defaults to 'plot'.
        Other option is 'save'.

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
        Z = rmg.at_node[surface].reshape(rmg.shape)
        # create list for non-axis (x_list)
        x_list = np.linspace(start_x, end_x, len(y_list))
        for x in range(0, len(x_list)):
            x_list[x] = np.rint(x_list[x])
        # use y_list and x_list to populate z_list
        z_list = []
        for i in range(0, len(y_list)):
            z_list.append(Z[y_list[i], x_list[i]])
        # plot(y_list, z_list)
        fig = plt.figure()
        plt.xlabel('Northing')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(y_list, z_list)
        plt.legend(loc='best')
    else:
        # create x_list
        x_list = list(range(start_x, end_x + 1, 1))
        num = len(x_list)
        ## create z_list
        # reshape surface
        Z = rmg.at_node[surface].reshape(rmg.shape)
        # create list for non-axis (y_list)
        y_list = np.linspace(start_y, end_y, num)
        for y in range(0, len(y_list)):
            y_list[y] = np.rint(y_list[y])
        # use x_list and y_list to populate z_list
        z_list = []
        for q in range(0, len(x_list)):
            z_list.append(Z[y_list[q], x_list[q]])
        # plot(x_list, z_list)
        fig = plt.figure()
        plt.xlabel('Easting')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(x_list, z_list)
        plt.legend(loc='best')
    plt.show()

def save_1D(rmg,
            surface,
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
    surface : field name
        Surface field name. Defaults to 'topographic__elevation' unless supplied.
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
        Z = rmg.at_node[surface].reshape(rmg.shape)
        # create list for non-axis (x_list)
        x_list = np.linspace(start_x, end_x, len(y_list))
        for x in range(0, len(x_list)):
            x_list[x] = np.rint(x_list[x])
        # use y_list and x_list to populate z_list
        z_list = []
        for i in range(0, len(y_list)):
            z_list.append(Z[y_list[i], x_list[i]])
        # plot(y_list, z_list)
        fig = plt.figure()
        plt.xlabel('Northing')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(y_list, z_list)
        plt.legend(loc='best')
    else:
        # create x_list
        x_list = list(range(start_x, end_x + 1, 1))
        num = len(x_list)
        ## create z_list
        # reshape surface
        Z = rmg.at_node[surface].reshape(rmg.shape)
        # create list for non-axis (y_list)
        y_list = np.linspace(start_y, end_y, num)
        for y in range(0, len(y_list)):
            y_list[y] = np.rint(y_list[y])
        # use x_list and y_list to populate z_list
        z_list = []
        for q in range(0, len(x_list)):
            z_list.append(Z[y_list[q], x_list[q]])
        # plot(x_list, z_list)
        fig = plt.figure()
        plt.xlabel('Easting')
        plt.ylabel('Elevation')
        plt.title(title)
        plt.plot(x_list, z_list)
        plt.legend(loc='best')
    plt.savefig("{}/{}.{}".format(output_location, tag, fileType), bbox_inches='tight')
    plt.close()

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

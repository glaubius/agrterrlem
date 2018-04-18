"""
output_lem.py

Functions to help with outputting data from LEM. Includes save_field_text to
save data in a Landlab Raster Model Grid field as a text file.

Written by Jennifer Glaubius, September 2017

"""
import numpy as np

def save_field_text(rmg, field, name, year, output_location, tag, file_type="txt"):
    """Save a field from rmg as text file (.txt).

    Parameters
    ----------
    rmg : RasterModelGrid
        RasterModelGrid created using Landlab containing topographic__elevation
        or other surface to be plotted.
    field : field name
        Surface field name.
    name : string
        Name for file.
    year : integer
        Simulated year.
    output_location : string
        Location file will be saved.
    tag : string
        Model run name.
    file_type : string
        Type of file format, e.g. .txt, .png. Defaults to txt
    """

    np.savetxt("{}{}-{:0>4}-{}.{}".format(output_location, name, year, tag, file_type), rmg.at_node[field], delimiter=",")

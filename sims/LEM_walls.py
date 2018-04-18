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

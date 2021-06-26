"""
===========
2_Anticipation_analysis.py
===========

These scripts produce the data for the plots of figure 4.
"""

import numpy as np

tmin_array = np.linspace(0,0.8,161)
tmax_array = np.round([x+0.01 for x in tmin_array],3)
tmin_seq = -0.2
tmax_seq = 0.8
decim = 2
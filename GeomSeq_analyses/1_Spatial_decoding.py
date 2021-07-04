"""
===========
1_Spatial_decoding
===========

The script produce the data used to plot figure 3.

Author: Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""

from GeomSeq_functions import spatial_decoding_funcs
from GeomSeq_analyses import config

# ======================================================================================================================
# ========== here are the different functions for the decoding of spatial location =====================================
# ======================================================================================================================

# Here choose the identifier of your subjects
subject = config.subjects_list[0]
# Generate GAT scores (Panel A): 4-folds cross validation of the angular decoding score
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,compute_cross_validation_score=True)

# Pannel B : apply the localizer to the 8 positions
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,compute_cross_validation_score=False)
spatial_decoding_funcs.apply_localizer_to_sequences_8positions(subject)

# Pannel C : apply the localizer built on the average of the data in the window 100-200 ms to the 8 positions of each sequences
# Create a localizer on the data averaged over the window from 100 to 200 ms. A sliding window of 25 corresponds to 100 ms (decim=4) window.
# The step value is just meant to be above the sliding window size.
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,tmin=0.1,tmax=0.2,SW=25,step=50)
spatial_decoding_funcs.apply_localizer_to_sequences(subject,classifier=False,tmin=0.1,tmax=0.2,SW=25,step=50)





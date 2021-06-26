"""
===========
2_Anticipation_analysis.py
===========

These scripts produce the data for the plots of figure 4.
"""

# build the classifier decoder data from the data on windows of 10 ms every 5 ms

import sys
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/GeomSeq/")
from GeomSeq_functions import spatial_decoding_funcs
from GeomSeq_analyses import config
from GeomSeq_functions import epoching_funcs


# 1 - create the classifier localizer decoder on the window 0 - 500ms on the data segmented
# every 5 ms on windows of 10 ms, decimation factor 2.
subject = config.subjects_list[0]
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=True,tmin = 0,tmax=0.5, SW=10,step = 5,decim=2)
spatial_decoding_funcs.apply_localizer_to_sequences(subject,classifier=True,tmin = -0.65,tmax=0.4, SW=10,step = 5,decim=2,
                                                    filter = "sequence != '4diagonal' and sequence != '2crosses'")

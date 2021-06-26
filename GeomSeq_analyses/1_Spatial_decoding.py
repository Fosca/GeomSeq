"""
===========
1_Spatial_decoding
===========

The script here will reproduce the results shown in Figure 3.
"""
import sys
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/GeomSeq/")

from GeomSeq_functions import spatial_decoding_funcs
from GeomSeq_analyses import config

# Here choose the identifier of your subjects
subject = config.subjects_list[0]
# Generate GAT scores (Panel A): 4-folds cross validation of the angular decoding score
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,compute_cross_validation_score=True)

# Pannel B : apply the localizer to the 8 positions
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,compute_cross_validation_score=False)
spatial_decoding_funcs.apply_localizer_to_sequences_8positions(subject)

# Pannel C : apply the localizer built on the average of the data in the window 100-200 ms to the 8 positions of each sequences
# Create a localizer on the data averaged over the window from 100 to 200 ms.
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,tmin=0.1,tmax=0.2,SW=25,step=50)
spatial_decoding_funcs.apply_localizer_to_sequences(subject,classifier=False,tmin=0.1,tmax=0.2,SW=25,step=50)





"""
===========
1_Spatial_decoding
===========

The script here will reproduce the results shown in Figure 3.
"""

from GeomSeq_functions import spatial_decoding_funcs
from GeomSeq_analyses import config


# Here choose the identifier of your subjects
subject = config.subjects_list[0]
# Generate GAT scores (Panel A): 4-folds cross validation of the angular decoding score
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,compute_cross_validation_score=True)

# Pannel B : apply the localizer to the sequences
spatial_decoding_funcs.build_localizer_decoder(subject,classifier=False,compute_cross_validation_score=False)




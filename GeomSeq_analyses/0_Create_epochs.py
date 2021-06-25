"""
===========
0_Create_epochs.py
===========

Script to epoch the data after preprocessing
"""


from GeomSeq_functions import epoching_funcs
from GeomSeq_analyses import config

# Here choose the identifier of your subjects
subject = config.subjects_list[0]
# epochs data from the primitive part of the experiment
epoching_funcs.compute_epochs(subject, tmin=-0.2, tmax=0.6, decim=4)
# from the sequence part of the experiment
epoching_funcs.compute_epochs(subject, tmin=-0.65, tmax=0.6, decim=4, block_type='sequences')
# from the localizer part of the experiment
epoching_funcs.compute_epochs(subject, tmin=-0.2, tmax=0.6, decim=4, block_type='localizer')





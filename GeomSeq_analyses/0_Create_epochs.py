"""
===========
0_Create_epochs.py
===========

Script to epoch the data after preprocessing

Author: Fosca Al Roumi <fosca.al.roumi@gmail.com>

"""

from GeomSeq_functions import epoching_funcs
from GeomSeq_analyses import config

# Here choose the identifier of your subjects
subject = config.subjects_list[0]
# epochs data from the primitive part of the experiment
epoching_funcs.compute_epochs(subject, tmin=-0.2, tmax=0.6, decim=1)
# from the sequence part of the experiment
epoching_funcs.compute_epochs(subject, tmin=-0.65, tmax=0.6, decim=1, block_type='sequences')
# from the localizer part of the experiment
epoching_funcs.compute_epochs(subject, tmin=-0.2, tmax=0.6, decim=1, block_type='localizer')

# from the sequence part of the experiment : epoch on the full sequence
epoching_funcs.compute_epochs(subject, tmin=0, tmax=0.433*8, decim=4, block_type='sequences',full_seq_block='full_seq')
# from the sequence part of the experiment : epoch on the full 12 repetitions of the sequences
epoching_funcs.compute_epochs(subject, tmin=0, tmax=0.433*8*12, decim=4, block_type='sequences',full_seq_block='full_block')



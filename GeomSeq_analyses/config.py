"""
===========
Config file
===========

Configuration parameters for the study.
"""

###############################################################################
# DIRECTORIES
# -----------
# Set the paths where your data, scripts etc are stored
root_path =  "/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/"
data_path = root_path + '/example_online/data/'
scripts_path = root_path + '/GeomSeq/'
result_path = root_path + '/example_online/results/'
figure_path =  root_path + '/example_online/figures/'

###############################################################################
# Put here the list of subjects and define the names of the runs you have chosen
# -----------
subjects_list = ['sub01','YYY']
runs = ['pairs1', 'pairs2', 'pairs3', 'pairs4', 'seq1', 'seq2', 'seq3',
        'seq4', 'localizer']
runs_dict = {subject: runs for subject in subjects_list}

ch_types = ['meg']
base_fname = '{extension}.fif'

shortest_event = 1
stim_channel = 'STI101'
min_event_duration = 0.002

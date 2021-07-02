"""
===========
4_Ordinal_knowledge_decoding
===========
The script produces the data of Figure 6, Figure S3, S4, S5, S6.
"""

import sys
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/GeomSeq/")
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/umne/")
from GeomSeq_analyses import config
from GeomSeq_functions import epoching_funcs, ordinal_decoders, utils

# ------- load the functions coming from our package --------
subject = config.subjects_list[0]
# --- il manque aussi le code pour générer la figure S6
# créer fonction qui fait epochs sur les séquences entières
# créer fonction qui extrait des runs les 12 répétitions pour chaque séquence: elle doit sortir 4 epochs contenant les 12 répétitions de chaque époque
# metadata doit contenir run_number et sequence




# ============== GAT data from ordinal decoding : Left columns of figures =======================================
ordinal_decoders.run_ordinal_decoding_GAT(subject)

# ============== Test the ordinal decoder from 300 to 500 ms on the full epoch presentation =======================================

# Middle column Figure 6
ordinal_decoders.decode_ordinal_position_oneSequence(subject,control=False)
# Middle column Figure S5
ordinal_decoders.decode_ordinal_position_oneSequence(subject,control=True)
# Middle column Figure S6
ordinal_decoders.decode_ordinal_position_oneSequence_CV(subject)
# Middle column Figure S4
ordinal_decoders.decode_ordinal_position_oneSequence_train24_test42(subject)

# ============== Test the ordinal decoder on the full block - cross validating across blocks =======================================

# Data for the right column Figure 6
ordinal_decoders.decode_ordinal_position_allBlocks(subject,control=False)
# Data for the right column Figure S5
ordinal_decoders.decode_ordinal_position_allBlocks(subject,control=True)
# Data for the right column Figure 6
ordinal_decoders.decode_ordinal_position_allBlocks_CV(subject, tmin=0.3, tmax=0.5,control=False)
# Data for the right column Figure S5
ordinal_decoders.decode_ordinal_position_allBlocks_CV(subject, tmin=0.3, tmax=0.5,control=False)
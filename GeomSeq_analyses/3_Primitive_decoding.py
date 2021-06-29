"""
===========
3_Primitive_decoding
===========
The script produce the data used to plot Figure 5.
"""
import sys
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/GeomSeq/")

from GeomSeq_functions import primitive_decoding_funcs
from GeomSeq_analyses import config

subject = config.subjects_list[0]

# ==== these are the functions to generate the results presented in Figure 5. ====
primitive_decoding_funcs.run_primitivepart_decoding_time_resolved(subject,which_primitives='11primitives',decim=4)
primitive_decoding_funcs.run_primitivepart_decoding_time_resolved(subject,which_primitives='rotVSsym',decim=4)
primitive_decoding_funcs.run_sequencepart_decoding_time_resolved(subject)



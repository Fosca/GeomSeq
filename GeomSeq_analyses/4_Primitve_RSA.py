"""
===========
4_RSA supplementary analysis
===========
The script produce the data used to plot Figure S2.

Author: Fosca Al Roumi <fosca.al.roumi@gmail.com>

"""
# ------- append the right paths --------

from GeomSeq_functions import rsa_funcs
from GeomSeq_analyses import config
import umne.scr.umne as umne
import numpy as np

# ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
subject = config.subjects_list[0]
rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
                                               which_analysis='primitives_and_sequences',
                                               factors_or_interest=('primitive', 'position_pair', 'block_type'))
# ______________________________________________________________________________________________________________________

# ---- extract the metadata of the empirical dissimilarity matrix -----
empir_dissim = np.load(config.result_path + '/rsa/dissim/primitives_and_sequencesprimitive_position_pair_block_type_no_baseline/spearmanr_sub01.dmat',allow_pickle=True)

# ====== now run the RSA regression analysis ==========

dis = rsa_funcs.dissimilarity_seq_prim
reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.result_path + 'rsa/dissim/primitives_and_sequencesprimitive_position_pair_block_type_no_baseline/*',
    [dis.block_type,dis.primitive,dis.primitive_different_blocks,dis.rotation_or_symmetry_psym, dis.rotation_or_symmetry_psym_different_blocks, dis.distance,
     dis.same_first, dis.same_second],
    included_cells_getter=None)

np.save(config.result_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',reg_dis)
reg_dis = np.load(config.result_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',allow_pickle=True)
times = [np.round(i, 0) for i in np.linspace(-0.4 * 1000, 1 * 1000, 131)]

# -- %% -- %% -- PLOTTING RSA PREDICTOR MATRICES AND REGRESSIONS -- %% -- %% --

#  --- Visualize the predictor matrices ---
def viz_predictor_mats(dissim_mat, min_val=0,max_val=1):
    umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dissim_mat, empir_dissim.md1),
                                get_label=lambda md: md['primitive'],min_val=min_val, max_value=max_val, plot_color_bar=False)

dis = rsa_funcs.dissimilarity_seq_prim
viz_predictor_mats(dis.block_type)
viz_predictor_mats(dis.primitive_different_blocks, min_val=-1,max_val=0)
viz_predictor_mats(dis.rotation_or_symmetry_psym)
viz_predictor_mats(dis.rotation_or_symmetry_psym_different_blocks, min_val=-1,max_val=0)
viz_predictor_mats(dis.same_first)
viz_predictor_mats(dis.same_second)
viz_predictor_mats(dis.distance)

# ----- plot all the regression coefficients on the same graph -----
fig = umne.rsa.plot_regression_results(reg_dis[:, :, :-1], times,
                                       legend=('Block_type','Primitive_ID','Primitive_ID_diff_blocks','Rotation_symmetry_psym','Rotation_symmetry_psym_diff_blocks', 'Distance', 'Same_first', 'Same_second'))

# ---- plot the regression coefficients separately ------
names = ('Block_type','Primitive_ID','Primitive_ID_diff_blocks','Rotation_symmetry_psym','Rotation_symmetry_psym_diff_blocks', 'Distance', 'Same_first', 'Same_second')
for ii, name in enumerate(names):
    fig = umne.rsa.plot_regression_results(reg_dis[:, :, ii, np.newaxis], times, ymin=-0.02, ymax=0.08)


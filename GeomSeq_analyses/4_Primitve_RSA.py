import sys
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/GeomSeq/")
sys.path.append("/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/umne/")

from GeomSeq_functions import rsa_funcs
import numpy as np

from GeomSeq_functions import config
import umne
import matplotlib.cm as cm

import matplotlib.pyplot as plt
from importlib import reload
from scipy.stats import ttest_1samp
from umne.stats import stats_cluster_based_permutation_test
reload(umne)
reload(rsa_funcs)
dis = rsa_funcs.dissimilarity
plt.close('all')
fig_path = '//Users/fosca/Documents/Fosca/Post_doc/Projects/Geom_seq/Article/Manuscript/revisions/new_figures/RSA/'

# ______________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________

# = = = = = = = = = = = = = = = 11 PRIMITIVES IN SEQUENCES AND PAIRS + POSITION PAIRS  = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# ______________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________


cc = np.load('/Volumes/COUCOU_CFC/GeomSeq_New/data/rsa/dissim/primitives_and_sequencesprimitive_position_pair_block_type_no_baseline/spearmanr_ml_180010.dmat',allow_pickle=True)

umne.rsaplot.video_dissim(cc, reordering='block_type_primitive_pair', which_labels='primitive', tmin=-0.4, tmax=1,
                          save_name='/Users/fosca/Desktop/videos/11prim_seq_prim')

dis = rsa_funcs.dissimilarity_seq_prim

# PREDICTOR MATRICES

# cc.md1, c'est les metadata de ta matrice de dissimilarité empirique

umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.block_type, cc.md1),
                            get_label=lambda md: md['primitive'], max_value=1, plot_color_bar=False)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'block_type.svg')


umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.primitive, cc.md1),
                            get_label=lambda md: md['primitive'], max_value=1, plot_color_bar=False)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'primitive_ID.svg')

umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.primitive_different_blocks, cc.md1),
                            get_label=lambda md: md['primitive'], min_val = -1, max_value=0, plot_color_bar=True)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'primitive_ID_diff_block.svg')

umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.rotation_or_symmetry_psym, cc.md1),
                            get_label=lambda md: md['primitive'], max_value=1, plot_color_bar=False)

plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'rotation_symm.svg')

umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.rotation_or_symmetry_psym_different_blocks, cc.md1),
                            get_label=lambda md: md['primitive'], min_val = -1, max_value=0, plot_color_bar=True)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'rotation_symm_diff_block.svg')


umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.same_first, cc.md1),
                            get_label=lambda md: md['primitive'], min_val = 0, max_value=1, plot_color_bar=True)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'same_first.svg')

umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.same_second, cc.md1),
                            get_label=lambda md: md['primitive'], min_val = 0, max_value=1, plot_color_bar=True)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'same_second.svg')

umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis.distance, cc.md1),
                            get_label=lambda md: md['primitive'], min_val = 0, max_value=1, plot_color_bar=True)
plt.gcf().savefig(fig_path + '/primitives_and_sequences_11primitives_and_position_pairs/theoretical_predictors/' + 'distance.svg')

# plt.gcf().savefig(fig_path + '/11primitives_presentedpairs/theoretical_predictors/' + 'primitive_ID.svg')


# =============== RSA 1 : WITHOUT THE GENERALIZATION STUFF =============
reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.saving_path + 'rsa/dissim/primitives_and_sequencesprimitive_position_pair_block_type_no_baseline/*',
    [dis.block_type,dis.primitive,dis.rotation_or_symmetry_psym, dis.distance,
     dis.same_first, dis.same_second],
    included_cells_getter=None)
# np.save(config.saving_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',reg_dis)
# reg_dis = np.load(config.saving_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',allow_pickle=True)
# times = [np.round(i, 0) for i in np.linspace(-0.4 * 1000, 1 * 1000, 131)]
times = [np.round(i, 0) for i in np.linspace(-0.4 * 1000, 1 * 1000, 131)]
fig = umne.rsa.plot_regression_results(reg_dis[:, :, :-1], times,
                                       legend=('Block_type','Primitive_ID','Rotation_symmetry_psym', 'Distance', 'Same_first', 'Same_second'))
names = ('Block_type','Primitive_ID','Rotation_symmetry_psym', 'Distance', 'Same_first', 'Same_second')
for ii, name in enumerate(names):
    fig = umne.rsa.plot_regression_results(reg_dis[:, :, ii, np.newaxis], times, ymin=-0.02, ymax=0.08)
    utils.create_folder(fig_path+"/primitives_and_sequences_11primitives_and_position_pairs_without_gene/regressions/")
    fig.savefig(fig_path+"/primitives_and_sequences_11primitives_and_position_pairs_without_gene/regressions/regression" + name + ".svg")

# =============== RSA 2 : JUST THE GENERALIZATION ACROSS BLOCKS =============

dis = rsa_funcs.dissimilarity_seq_prim

def get_different_blocks(matrix):
    md0 = matrix.md0
    md1 = matrix.md1
    def include(i, j):
        return md0['block_type'][i] != md1['block_type'][j]
    return [(i, j) for i in range(len(md0)) for j in range(len(md1)) if include(i, j)]

reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.saving_path + 'rsa/dissim/primitives_and_sequencesprimitive_position_pair_block_type_no_baseline/*',
    [dis.primitive_different_blocks, dis.rotation_or_symmetry_psym_different_blocks, dis.distance,
     dis.same_first, dis.same_second],
    included_cells_getter=get_different_blocks)
# np.save(config.saving_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',reg_dis)
# reg_dis = np.load(config.saving_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',allow_pickle=True)
# times = [np.round(i, 0) for i in np.linspace(-0.4 * 1000, 1 * 1000, 131)]
times = [np.round(i, 0) for i in np.linspace(-0.4 * 1000, 1 * 1000, 131)]
fig = umne.rsa.plot_regression_results(reg_dis[:, :, :-1], times,
                                       legend=(
                                       'Primitive_ID_diff_blocks', 'Rotation_symmetry_psym_diff_blocks', 'Distance', 'Same_first',
                                       'Same_second'))
names = ('Primitive_ID_diff_blocks', 'Rotation_symmetry_psym_diff_blocks', 'Distance', 'Same_first', 'Same_second')
for ii, name in enumerate(names):
    fig = umne.rsa.plot_regression_results(reg_dis[:, :, ii, np.newaxis], times, ymin=-0.02, ymax=0.08)
    utils.create_folder(
        fig_path + "/primitives_and_sequences_11primitives_and_position_pairs_without_gene/regressions/")
    fig.savefig(
        fig_path + "/primitives_and_sequences_11primitives_and_position_pairs_without_gene/regressions/regression_diffblock" + name + ".svg")

# =============== RSA 3 : Everything together =============
dis = rsa_funcs.dissimilarity_seq_prim

reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.saving_path + 'rsa/dissim/primitives_and_sequencesprimitive_position_pair_block_type_no_baseline/*',
    [dis.block_type,dis.primitive,dis.primitive_different_blocks,dis.rotation_or_symmetry_psym, dis.rotation_or_symmetry_psym_different_blocks, dis.distance,
     dis.same_first, dis.same_second],
    included_cells_getter=None)

np.save(config.saving_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',reg_dis)
reg_dis = np.load(config.saving_path + 'rsa/regressions/pairs_and_sequences/regression_no_baseline.npy',allow_pickle=True)
times = [np.round(i, 0) for i in np.linspace(-0.4 * 1000, 1 * 1000, 131)]

fig = umne.rsa.plot_regression_results(reg_dis[:, :, :-1], times,
                                       legend=('Block_type','Primitive_ID','Primitive_ID_diff_blocks','Rotation_symmetry_psym','Rotation_symmetry_psym_diff_blocks', 'Distance', 'Same_first', 'Same_second'))



names = ('Block_type','Primitive_ID','Primitive_ID_diff_blocks','Rotation_symmetry_psym','Rotation_symmetry_psym_diff_blocks', 'Distance', 'Same_first', 'Same_second')
for ii, name in enumerate(names):
    fig = umne.rsa.plot_regression_results(reg_dis[:, :, ii, np.newaxis], times, ymin=-0.02, ymax=0.08)
    # fig.patch.set_visible(False)
    # plt.gca().axis('off')
    fig.savefig(fig_path+"/primitives_and_sequences_11primitives_and_position_pairs/regressions/regression" + name + ".svg")



dissim_block_type = rsa_funcs.gen_predicted_dissimilarity(dis.block_type,md = cc.md1)
dissim_prim = rsa_funcs.gen_predicted_dissimilarity(dis.primitive,md = cc.md1)
dissim_prim_diff_blocks = rsa_funcs.gen_predicted_dissimilarity(dis.primitive_different_blocks,md = cc.md1)
dissim_rotorsym = rsa_funcs.gen_predicted_dissimilarity(dis.rotation_or_symmetry,md = cc.md1)
dissim_rotorsym_diff_blocks = rsa_funcs.gen_predicted_dissimilarity(dis.rotation_or_symmetry_different_blocks,md = cc.md1)
dissim_distance = rsa_funcs.gen_predicted_dissimilarity(dis.distance,md = cc.md1)
dissim_samefirst = rsa_funcs.gen_predicted_dissimilarity(dis.same_first,md = cc.md1)
dissim_samesecond = rsa_funcs.gen_predicted_dissimilarity(dis.same_second,md = cc.md1)


dissim_matrix = [dissim_block_type,dissim_prim,dissim_prim_diff_blocks,dissim_rotorsym,dissim_rotorsym_diff_blocks,dissim_distance,dissim_samefirst,dissim_samesecond]
correlation_matrix = np.zeros((8,8))

for k in range(8):
    for l in range(8):
        r = np.corrcoef([np.reshape(dissim_matrix[k].data, dissim_matrix[k].data.size),
                         np.reshape(dissim_matrix[l].data, dissim_matrix[l].data.size)])
        correlation_matrix[k,l]=r[0,1]

plt.imshow(correlation_matrix, cmap=cm.viridis)
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(8),names,rotation=30)
plt.yticks(range(8),names,rotation=30)

fig = plt.gcf()
fig.savefig('correlation_regressors.png')

# =================================== STATS STATS STATS STATS STATS ================================

inter_stim_times = np.where(np.logical_and(np.asarray(times)>0,(np.asarray(times)<433)))[0]

pval = stats_cluster_based_permutation_test(reg_dis[:,:,1])
np.asarray(times)[pval<0.05]

after_first = np.where(np.asarray(times)>0)[0]
pval_after = stats_cluster_based_permutation_test(reg_dis[:,after_first,1])
np.asarray(times)[after_first][pval_after<0.05]

before_first = np.where(np.asarray(times)<=0)[0]
pval_before = stats_cluster_based_permutation_test(reg_dis[:,before_first,1])
np.asarray(times)[before_first][pval_before<0.05]


# ===== block type ======
pval_blocktype = stats_cluster_based_permutation_test(reg_dis[:,:,0])
np.asarray(times)[pval_blocktype<0.05]


# ===== distance ======
pval_samedist = stats_cluster_based_permutation_test(reg_dis[:,:,-4])
np.asarray(times)[pval_samedist<0.05]

# ===== same first ======
pval_samefirst = stats_cluster_based_permutation_test(reg_dis[:,:,-3])
np.asarray(times)[pval_samefirst<0.05]

# ===== same second ======
pval_samesecond = stats_cluster_based_permutation_test(reg_dis[:,:,-2])
np.asarray(times)[pval_samesecond<0.05]

# ===== operation type ======
mean_rotsym = np.mean(reg_dis[:,:,3],axis=1)
t,p = ttest_1samp(mean_rotsym,popmean=0)



sts = ttest_1samp(np.mean(reg_dis[:,inter_stim_times,3],axis=1),0)
print(sts)
selected_times = np.asarray(times)[inter_stim_times]

pval = stats_cluster_based_permutation_test(reg_dis[:,after_first,3])
selected_times[pval<0.05]

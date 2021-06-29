
from GeomSeq_functions import epoching_funcs, utils
from GeomSeq_analyses import config
import pandas as pd
import mne
import numpy as np
import umne
import glob
import os.path as op

class fn_template:
    epochs_full = config.data_path + "rsa/rsa_epochs/full-{:}/epochs_{:}.fif"
    dissim = config.result_path + "rsa/dissim/{:}/{:}_{:}.dmat"

#-----------------------------------------------------------------------------------------------------------------------
def preprocess_and_compute_dissimilarity(subject, metrics, tmin=-0.4, tmax=1.,decim=1,
                                         baseline=(None, 0), rejection='default',
                                         which_analysis='primitives_and_sequences',
                                         factors_or_interest = ('run_number', 'primitive','position_pair','block_type')):
    """
    We compute the empirical dissimilarity for the data averaging the epochs across the factors of interest
    :param subject:
    :param metrics: The distance metric for the notion of similarity.
    The following parameters are parameters for the epoching:
    :param tmin:
    :param tmax:
    :param decim:
    :param baseline:
    :param rejection:
    :param which_analysis: 'primitives', 'sequences' or 'primitives_and_sequences'
    :param factors_or_interest: The factors across which the epochs are averaged to compute the dissimilarity
    :return:
    """
    if isinstance(metrics, str):
        metrics = metrics,

    reject = None
    if rejection=='default':
        reject = dict(
            grad=4000e-13,
            mag=4e-12)

    if baseline is None:
        bl_str = '_no_baseline'
    else:
        bl_str = '_baseline'

    epochs_RSA = extract_good_epochs_for_RSA(subject, tmin, tmax, baseline, decim, reject, which_analysis)

    # ========= split half method ================
    epochs_1 = epochs_RSA["run_number == 1 or run_number == 3"]
    epochs_2 = epochs_RSA["run_number == 2 or run_number == 4"]

    avg_1 = umne.epochs.average_epochs_by_metadata(epochs_1, factors_or_interest)
    avg_2 = umne.epochs.average_epochs_by_metadata(epochs_2, factors_or_interest)

    del epochs_1
    del epochs_2

    for metric in metrics:
        _compute_and_save_dissimilarity(avg_1, avg_2, which_analysis + '_'.join(factors_or_interest) + bl_str, subject, metric)

    print('Saved.')


#-----------------------------------------------------------------------------------------------------------------------
def extract_good_epochs_for_RSA(subject,tmin,tmax,baseline,decim,reject,which_analysis):
    """
    This function computes and saves the epochs epoched for the RSA.
    :param subject:
    :param tmin:
    :param tmax:
    :param baseline:
    :param decim:
    :param reject:
    :param which_analysis:
    :return:
    """

    epochs = []
    if 'primitives' in which_analysis:
        epoching_funcs.compute_epochs(subject, tmin, tmax, baseline=baseline, decim=decim, suffix="rsa", reject=reject)
        epochs_pairs = epoching_funcs.load_and_concatenate_epochs(subject, suffix="rsa*")
        epochs_p = epochs_pairs["first_or_second == 1 and violation == 0 and primitive != 'control'"]
        epochs.append(epochs_p)
    if 'sequences' in which_analysis:
        epoching_funcs.compute_epochs(subject, tmin, tmax, baseline=baseline, decim=decim, suffix="rsa", reject=reject,block_type='sequences')
        epochs_seqs = epoching_funcs.load_and_concatenate_epochs(subject, suffix ="rsa*", filter='seq')
        epochs_s = epochs_seqs["sequence != 'memory1' and sequence != 'memory2' and sequence != 'memory4' and sequence != 'irregular' and primitive_level1 != 'nan' and position_pair != 'nan' and violation_primitive == 0"]
        epochs.append(epochs_s)
    epochs = mne.concatenate_epochs(epochs)

    return epochs



#-----------------------------------------------------------------------------------------------------------------------
def _compute_and_save_dissimilarity(epochs1, epochs2, subdir, subj_id, metric):

    print('\n\nComputing {:} dissimilarity (metric={:})...'.format(subdir, metric))

    dissim = umne.rsa.gen_observed_dissimilarity(epochs1, epochs2, metric=metric, sliding_window_size=100, sliding_window_step=10)

    filename = fn_template.dissim.format(subdir, metric, subj_id)
    utils.create_folder(op.split(filename)[0]+'/')
    print('Saving the dissimilarity matrix to {:}'.format(filename))
    dissim.save(filename)



# ======================================================================================================================
# ======================================================================================================================

#-----------------------------------------------------------------------------------------------------------------------
class dissimilarity:
    """
    Target dissimilarity functions
    Each function gets two dictionnaries containing several metadata fields and returns a dissimilarity score (high = dissimilar)
    """

    # ---------------------------------------------------------
    @staticmethod
    def run_number(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """
        #-- Array indicating to which run the trial belongs
        run_number1 = stim1['run_number']
        run_number2 = stim2['run_number']

        return 0 if run_number1 == run_number2 else 1

    @staticmethod
    def sub_run(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """
        #-- Array indicating to which run the trial belongs
        run_number1 = stim1['run_number']
        prim1 = stim1['primitive']
        run_number2 = stim2['run_number']
        prim2 = stim2['primitive']


        return 0 if (run_number1 == run_number2 and prim1 == prim2) else 1


    # ---------------------------------------------------------
    @staticmethod
    def primitive(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar

        """
        prim1 = stim1['primitive']
        prim2 = stim2['primitive']

        return 0 if prim1 == prim2 else 1


    #---------------------------------------------------------
    @staticmethod
    def same_first(stim1, stim2):
        """
        Is the position of the first item the same ?
        """
        f_all = []
        for f in [stim1['position_pair'],stim2['position_pair']]:
            if len(str(f))==1:
                f = '0'+str(f)
            else:
                f = str(f)
            f_all.append(f)


        return 0 if f_all[0][0] == f_all[1][0] else 1


    #---------------------------------------------------------
    @staticmethod
    def same_second(stim1, stim2):
        """
        Is the position of the second item the same ?
        """
        f_all = []
        for f in [stim1['position_pair'],stim2['position_pair']]:
            if len(str(f))==1:
                f = '0'+str(f)
            else:
                f = str(f)
            f_all.append(f)


        return 0 if f_all[0][1] == f_all[1][1] else 1

    #---------------------------------------------------------
    @staticmethod
    def distance(stim1, stim2):
        """
        Is the distance involved in the first pair of points the same as the second ?
        """
        lengths = [0,1,np.sqrt(2 + np.sqrt(2)),1 + np.sqrt(2),np.sqrt(4 + 2*np.sqrt(2)),1 + np.sqrt(2),np.sqrt(2 + np.sqrt(2)),1]

        f_all = []
        for f in [stim1['position_pair'],stim2['position_pair']]:
            if len(str(f))==1:
                f = '0'+str(f)
            else:
                f = str(f)
            f_all.append(f)


        two_pos1 = [int(f_all[0][0]),int(f_all[0][1])]
        two_pos2 = [int(f_all[1][0]),int(f_all[1][1])]

        distance1 = lengths[int(np.max(two_pos1)-np.min(two_pos1))]
        distance2 = lengths[int(np.max(two_pos2)-np.min(two_pos2))]

        return np.abs(distance2 - distance1)

    #---------------------------------------------------------
    @staticmethod
    def rotation_or_symmetry(stim1, stim2):
        """
        """

        isrot1 = 'rot' in stim1['primitive']
        isrot2 = 'rot' in stim2['primitive']
        issym1 = any(substring in stim1['primitive'] for substring in ['A','B','H','V'])
        issym2 = any(substring in stim2['primitive'] for substring in ['A','B','H','V'])

        if isrot1 and isrot2:
            a =  0
        elif issym1 and issym2:
            a = 0
        else:
            a =  1
        return a

    #---------------------------------------------------------
    @staticmethod
    def rotation_or_symmetry_psym(stim1, stim2):
        """
        """
        isrot1 = 'rot' in stim1['primitive']
        isrot2 = 'rot' in stim2['primitive']
        issym1 = any(substring in stim1['primitive'] for substring in ['A','B','H','V','sym_point'])
        issym2 = any(substring in stim2['primitive'] for substring in ['A','B','H','V','sym_point'])

        if isrot1 and isrot2:
            a =  0
        elif issym1 and issym2:
            a = 0
        else:
            a =  1
        return a
    #---------------------------------------------------------
    @staticmethod
    def block_type(stim1, stim2):
        """
        """
        isseq_block1 = 'seq' in stim1['block_type']
        isseq_block2 = 'seq' in stim2['block_type']

        return 0 if isseq_block1 == isseq_block2  else 1


# ================================================================================================================
class dissimilarity_seq_prim:
    """
    Target dissimilarity functions
    in the case we are analyzing together the primitive and the sequence blocks
    """

    # ---------------------------------------------------------
    @staticmethod
    def primitive(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar

        """
        prim1 = stim1['primitive']
        prim2 = stim2['primitive']
        block_type1 = stim1['block_type']
        block_type2 = stim2['block_type']

        return 0 if np.logical_and(prim1 == prim2,block_type1==block_type2) else 1

    # ---------------------------------------------------------
    @staticmethod
    def primitive_different_blocks(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar

        """
        prim1 = stim1['primitive']
        prim2 = stim2['primitive']
        block_type1 = stim1['block_type']
        block_type2 = stim2['block_type']

        return 0 if np.logical_and(prim1 == prim2, block_type1 != block_type2) else 1

    #---------------------------------------------------------
    @staticmethod
    def same_first(stim1, stim2):
        """
        Is the position of the first item the same ?
        """
        f_all = []
        for f in [stim1['position_pair'],stim2['position_pair']]:
            if len(str(f))==1:
                f = '0'+str(f)
            else:
                f = str(f)
            f_all.append(f)


        return 0 if f_all[0][0] == f_all[1][0] else 1


    #---------------------------------------------------------
    @staticmethod
    def same_second(stim1, stim2):
        """
        Is the position of the second item the same ?
        """
        f_all = []
        for f in [stim1['position_pair'],stim2['position_pair']]:
            if len(str(f))==1:
                f = '0'+str(f)
            else:
                f = str(f)
            f_all.append(f)


        return 0 if f_all[0][1] == f_all[1][1] else 1

    #---------------------------------------------------------
    @staticmethod
    def distance(stim1, stim2):
        """
        Is the distance involved in the first pair of points the same as the second ?
        """
        lengths = [0,1,np.sqrt(2 + np.sqrt(2)),1 + np.sqrt(2),np.sqrt(4 + 2*np.sqrt(2)),1 + np.sqrt(2),np.sqrt(2 + np.sqrt(2)),1]

        f_all = []
        for f in [stim1['position_pair'],stim2['position_pair']]:
            if len(str(f))==1:
                f = '0'+str(f)
            else:
                f = str(f)
            f_all.append(f)


        two_pos1 = [int(f_all[0][0]),int(f_all[0][1])]
        two_pos2 = [int(f_all[1][0]),int(f_all[1][1])]

        distance1 = lengths[int(np.max(two_pos1)-np.min(two_pos1))]
        distance2 = lengths[int(np.max(two_pos2)-np.min(two_pos2))]

        return np.abs(distance2 - distance1)

    #---------------------------------------------------------
    @staticmethod
    def rotation_or_symmetry(stim1, stim2):
        """
        """

        isrot1 = 'rot' in stim1['primitive']
        isrot2 = 'rot' in stim2['primitive']
        issym1 = any(substring in stim1['primitive'] for substring in ['A','B','H','V'])
        issym2 = any(substring in stim2['primitive'] for substring in ['A','B','H','V'])
        block_type1 = stim1['block_type']
        block_type2 = stim2['block_type']


        if isrot1 and isrot2:
            a =  0
        elif issym1 and issym2:
            a = 0
        else:
            a =  1

        if block_type1 != block_type2:
            a = 1

        return a

    #---------------------------------------------------------
    @staticmethod
    def rotation_or_symmetry_psym(stim1, stim2):
        """
        """
        isrot1 = 'rot' in stim1['primitive']
        isrot2 = 'rot' in stim2['primitive']
        issym1 = any(substring in stim1['primitive'] for substring in ['A','B','H','V','sym_point'])
        issym2 = any(substring in stim2['primitive'] for substring in ['A','B','H','V','sym_point'])
        block_type1 = stim1['block_type']
        block_type2 = stim2['block_type']


        if isrot1 and isrot2:
            a =  0
        elif issym1 and issym2:
            a = 0
        else:
            a =  1

        if block_type1 != block_type2:
            a = 1

        return a

    #---------------------------------------------------------
    @staticmethod
    def rotation_or_symmetry_different_blocks(stim1, stim2):
        """
        """

        isrot1 = 'rot' in stim1['primitive']
        isrot2 = 'rot' in stim2['primitive']
        issym1 = any(substring in stim1['primitive'] for substring in ['A','B','H','V'])
        issym2 = any(substring in stim2['primitive'] for substring in ['A','B','H','V'])

        block_type1 = stim1['block_type']
        block_type2 = stim2['block_type']

        if isrot1 and isrot2:
            a =  0
        elif issym1 and issym2:
            a = 0
        else:
            a =  1

        if block_type1 == block_type2:
            a = 1

        return a

    #---------------------------------------------------------
    @staticmethod
    def rotation_or_symmetry_psym_different_blocks(stim1, stim2):
        """
        """
        isrot1 = 'rot' in stim1['primitive']
        isrot2 = 'rot' in stim2['primitive']
        issym1 = any(substring in stim1['primitive'] for substring in ['A','B','H','V','sym_point'])
        issym2 = any(substring in stim2['primitive'] for substring in ['A','B','H','V','sym_point'])
        block_type1 = stim1['block_type']
        block_type2 = stim2['block_type']

        if isrot1 and isrot2:
            a =  0
        elif issym1 and issym2:
            a = 0
        else:
            a =  1

        if block_type1 == block_type2:
            a = 1


        return a


    #---------------------------------------------------------
    @staticmethod
    def block_type(stim1, stim2):
        """
        """
        isseq_block1 = 'seq' in stim1['block_type']
        isseq_block2 = 'seq' in stim2['block_type']

        return 0 if isseq_block1 == isseq_block2  else 1

#-----------------------------------------------------------------------------------------------------------------------
def get_top_triangle_inds(matrix):
    md0 = matrix.md0
    md1 = matrix.md1

    def include(i, j):
        return md0['target'][i] > md1['target'][j] or \
               (md0['target'][i] == md1['target'][j] and md0['location'][i] > md1['location'][j])

    return [(i, j) for i in range(len(md0)) for j in range(len(md1)) if include(i, j)]

#-----------------------------------------------------------------------------------------------------------------------
def all_stimuli():

    # ====== we load the metadata from a given participant ==========
    metadata_path = config.data_path+'rsa/all_stimuli.pkl'
    all_stimuli = pd.read_pickle(metadata_path)
    all_stimuli = all_stimuli[np.logical_and(all_stimuli['first_or_second']==1,all_stimuli['violation']==0)]

    all_dict = []
    for primitive in ['rotp1','rotm1','rotp2','rotm2','rotp3','rotm3','sym_point','A','B','H','V']:
        presented_pairs = np.unique(all_stimuli[all_stimuli['primitive']==primitive]['position_pair'])
        for k in range(len(presented_pairs)):
            pres_pair = presented_pairs[k]
            for run_numb in range(2,6):
                all_dict.append(pd.DataFrame([dict(primitive=primitive,position_pair=pres_pair,run_number=run_numb)]))

    df = pd.concat(all_dict)

    return df

#-----------------------------------------------------------------------------------------------------------------------
def gen_predicted_dissimilarity(dissimilarity_func,md=None):
    """
    Generate a predicted dissimilarity matrix (for all stimuli)
    """
    if md is None:
        md = all_stimuli()

    result = umne.rsa.gen_predicted_dissimilarity(dissimilarity_func, md, md)

    return umne.rsa.DissimilarityMatrix([result], md, md)

#-----------------------------------------------------------------------------------------------------------------------
def reshape_matrix_2(dissimilarity_matrix,fields =('primitive','position_pair')):
    """
    The goal of this function is to reshape the dissimilarity matrix. The goal is ultimately to average all the dissimilarity matrices.
    For this function, all the participants should have the same metadata of interest.
    :param diss_mat:
    :param reshape_order: the list of fields that says in which hierarchical order we want to organize the data
    :return:
    """
    meta_original = dissimilarity_matrix.md0
    mapping = {key:[] for key in fields}
    indices = {'initial_index':[],'final_index':[]}

    meta_filter = meta_original.copy()

    counter = 0
    key_values1 = np.unique(meta_original[fields[0]])
    for val1 in key_values1:
        meta_filter1 = meta_original[meta_filter[fields[0]].values == val1]
        key_values2 = np.unique(meta_filter1[fields[1]])
        for val2 in key_values2:
            meta_filter2 = meta_filter1[meta_filter1[fields[1]].values == val2]
            idx = meta_filter2.index[0]
            indices['initial_index'].append(idx)
            indices['final_index'].append(counter)
            mapping[fields[0]].append(val1)
            mapping[fields[1]].append(val2)
            counter += 1

    dissim_final = np.nan*np.ones((dissimilarity_matrix.data.shape[0],counter,counter))

    for m in range(counter):
        ind_m = indices['initial_index'][m]
        if ind_m is not None:
            for n in range(counter):
                ind_n = indices['initial_index'][n]
                if ind_n is not None:
                            dissim_final[:,m,n] = dissimilarity_matrix.data[:,ind_m,ind_n]

    meta_final = pd.DataFrame.from_dict(mapping)

    dissimilarity_matrix.data = dissim_final
    dissimilarity_matrix.md0 = meta_final
    dissimilarity_matrix.md1 = meta_final

    return dissimilarity_matrix

#-----------------------------------------------------------------------------------------------------------------------
def put_in_good_shape(dissimilarity_matrix,block_types = ['primitives','sequences']):
    """
    The goal of this function is to put all the dissimilarity matrices of all the participants in the same shape in order
    to later average them.
    :param diss_mat:
    :return:
    """
    all_stims = all_stimuli()

    meta_original = dissimilarity_matrix.md0
    res_map = dict(block_type=[],primitives=[],presented_pairs=[],run_number=[],presence_data=[],initial_index= [],final_index = [])

    prim_analysis = 0
    if block_types == ['primitives']:
        print('There is no metadata field when only the primitive analysis was ran')
        prim_analysis = 1

    counter = 0
    for block in block_types:
        for prim in np.unique(all_stims['primitive'].values):
            pairs = np.unique(all_stims.query("primitive == '%s'"%prim)['position_pair'].values)
            for pair in pairs:
                for run in range(2,6):
                    if prim_analysis:
                        is_there_something = meta_original.query("primitive == '%s' and position_pair == %i and run_number == %i"%(prim, pair, run))
                    else:
                        is_there_something = meta_original.query("block_type=='%s' and primitive == '%s' and position_pair == %i and run_number == %i"%(block,prim, pair, run))

                    if len(is_there_something)==1 :
                        idx = is_there_something.index[0]
                        res_map['presence_data'].append(1)
                        res_map['initial_index'].append(idx)
                        res_map['final_index'].append(counter)
                    else:
                        res_map['presence_data'].append(0)
                        res_map['initial_index'].append(None)
                        res_map['final_index'].append(counter)

                    res_map['block_type'].append(block)
                    res_map['primitives'].append(prim)
                    res_map['presented_pairs'].append(pair)
                    res_map['run_number'].append(run)

                    counter +=1

    dissim_final = np.nan*np.ones((dissimilarity_matrix.data.shape[0],counter,counter))

    # And now we fill the array

    for m in range(counter):
        ind_m = res_map['initial_index'][m]
        if ind_m is not None:
            for n in range(counter):
                ind_n = res_map['initial_index'][n]
                if ind_n is not None:
                            dissim_final[:,m,n] = dissimilarity_matrix.data[:,ind_m,ind_n]

    meta_final = pd.DataFrame.from_dict(dict(primitive=res_map['primitives'],position_pair=res_map['presented_pairs'],run_number=res_map['run_number'],block_type = res_map['block_type']))

    dissimilarity_matrix.data = dissim_final
    dissimilarity_matrix.md0 = meta_final
    dissimilarity_matrix.md1 = meta_final

    return dissimilarity_matrix

#-----------------------------------------------------------------------------------------------------------------------
def load_and_avg_dissimilarity_matrices(analysis_type_path,block_types=['primitives'],keep_initial_shape=False,fields=None):


    files = glob.glob(analysis_type_path+'/*')
    print(files)
    diss_all = []

    for file in files:
        dissimilarity_matrix = np.load(file,allow_pickle=True)
        print(file)
        if keep_initial_shape:
            print("We keep the initial shape")
            print("The shape of the data is ")
            print(dissimilarity_matrix.data.shape)
        else:
            print("The initial shape of the data is ")
            print(dissimilarity_matrix.data.shape)
            if fields is not None:
                if len(fields)==2:
                    dissimilarity_matrix = reshape_matrix_2(dissimilarity_matrix,fields=fields)
            else:
                dissimilarity_matrix = put_in_good_shape(dissimilarity_matrix,block_types=block_types)
            print("The final shape of the data is ")
            print(dissimilarity_matrix.data.shape)
        diss_all.append(dissimilarity_matrix.data)

    diss_all = np.asarray(diss_all)
    dissimilarity_matrix.data = diss_all
    return dissimilarity_matrix
















from sklearn.model_selection import StratifiedKFold
from GeomSeq_analyses import config
from GeomSeq_functions import epoching_funcs
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import GeneralizingEstimator
import mne
# ______________________________________________________________________________________________________________________
def balance_presented_pairs(epochs_pairs,block_type="pairs"):
    """
    This function will output epochs where transformations are symmetries and rotations that involve the
    exact same locations on the screen.
    :param epochs_pairs: These can come from the sequence or the primitive part
    :param block_type: The type of block. 'pairs' if from primitive part, 'seq' if from sequence part and
    'seq_macaque' if from the macaque data.
    :return: The epochs object with balanced presented pairs.
    """
    if block_type=="pairs":
        epochs_pairs = epochs_pairs["first_or_second == 1 and violation == 0"]
        epochs_sym = epochs_pairs["rotation_or_symmetry == 'symmetry'"]
        epochs_rot = epochs_pairs["rotation_or_symmetry == 'rotation'"]
    elif 'seq' in block_type:
        print("========= we are dealing with trials from the sequence part =========")
        if 'maca' in block_type:
            epochs_pairs = epochs_pairs["primitive_level1 != 'nan'"]
        else:
            epochs_pairs = epochs_pairs["primitive_level1 != 'nan' and violation == 0"]
        epochs_sym = epochs_pairs["primitive_level1 == 'A' or primitive_level1 == 'B' or primitive_level1 == 'H' or primitive_level1 == 'V'"]
        epochs_sym.metadata["rotation_or_symmetry"] = np.asarray(['symmetry']*len(epochs_sym))
        epochs_rot = epochs_pairs["primitive_level1 == 'rotp1' or primitive_level1 == 'rotm1' or primitive_level1 == 'rotp3' or primitive_level1 == 'rotm3'"]
        epochs_rot.metadata["rotation_or_symmetry"] = np.asarray(['rotation'] * len(epochs_rot))
    else:
        print("Error in the block type")

    pairs_sym = epochs_sym.metadata["position_pair"].values
    pairs_rot = epochs_rot.metadata["position_pair"].values

    pairs_rot_balanced = []
    pairs_sym_balanced = []

    for pres_pair in np.unique(pairs_sym):
        inds_sym = np.where(pairs_sym==pres_pair)[0]
        inds_rot = np.where(pairs_rot==pres_pair)[0]
        min_numb = np.min([len(inds_sym),len(inds_rot)])
        inds_sym = np.random.choice(inds_sym,min_numb,replace=False)
        inds_rot = np.random.choice(inds_rot,min_numb,replace=False)
        if min_numb !=0:
            pairs_sym_balanced.append(epochs_sym[inds_sym])
            pairs_rot_balanced.append(epochs_rot[inds_rot])

    pairs_rot_balanced = mne.concatenate_epochs(pairs_rot_balanced)
    pairs_sym_balanced = mne.concatenate_epochs(pairs_sym_balanced)

    np.unique(pairs_rot_balanced.metadata["primitive"])

    return mne.concatenate_epochs([pairs_rot_balanced,pairs_sym_balanced])


def gat_classifier_categories(epochs,labels,inds_train,inds_test,micro_avg_or_not,scoring=None,predict_probability=False,n_jobs=8):

    """
    Run a temporal generalization SVM decoder
    :param epochs: epochs on which data we decode
    :param labels: labels for decoding
    :param inds_train: list of indices for the training
    :param inds_test: list of indices for the testing
    :param micro_avg_or_not:
    :param scoring: Scoring method
    :return: The scores for the different folds
    """

    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability = predict_probability))
    time_gen = GeneralizingEstimator(clf, scoring=scoring, n_jobs=n_jobs, verbose=True)

    y_preds_probas = []
    metadata_train = []
    metadata_test = []
    y_true = []
    scores = []
    distances = []
    y_preds = []

    for cv in range(len(inds_train)):

        train = inds_train[cv]
        test = inds_test[cv]
        X_train = epochs[train].get_data()
        meta_train = epochs[train].metadata
        metadata_train.append(meta_train)
        X_test = epochs[test].get_data()
        meta_test = epochs[test].metadata
        metadata_test.append(meta_test)
        y_train = labels[train]
        y_test = labels[test]

        # ==================================== When performing micro-averaging ==============================================
        if micro_avg_or_not:
            X_train_micro = []
            y_train_micro = []
            X_test_micro = []
            y_test_micro = []

            for lab in np.unique(labels):
                print('===== we are micro averaging the trials for condition %i ===='%lab)
                epo_train = epochs[train]
                epo_train_lab = epo_train[y_train==lab]
                X_train_micro.append(micro_avg(epo_train_lab, navg=5))
                y_train_micro.append([lab]*np.sum(y_train==lab))
                epo_test = epochs[test]
                epo_test_lab = epo_test[y_test==lab]
                X_test_micro.append(micro_avg(epo_test_lab, navg=5))
                y_test_micro.append([lab]*np.sum(y_test==lab))
            X_train = np.vstack(X_train_micro)
            y_train = np.hstack(y_train_micro)
            X_test = np.vstack(X_test_micro)
            y_test = np.hstack(y_test_micro)

        # ==================================== MICROAVERAGING PARENTHESIS ==============================================
        time_gen.fit(X=X_train,
                     y=y_train)
        y_preds.append(time_gen.predict(X=X_test))
        scores.append(time_gen.score(X=X_test,
                     y=y_test))
        y_true.append(y_test)

        if predict_probability:
            y_preds_probas.append(time_gen.predict_proba(X=X_test))
            distances.append(time_gen.decision_function(X=X_test))

    if predict_probability:
        return dict(scores=scores,y_preds_probas=y_preds_probas,y_true=y_true,meta_train=metadata_train,meta_test=metadata_test,distances=distances), epochs.times

    return np.asarray(scores),epochs.times

# ===========================================================================================
def micro_avg(epochs, navg=5):

    """
    This function builds the micro-averaged data. If we start with n epochs, we get n micro-averaged epochs
    :param epochs:
    :param navg:
    :return: micro-averaged data
    """

    data = epochs.get_data()
    n_epochs = data.shape[0]
    idx = np.asarray(list(range(n_epochs))*navg)
    # ======= make sure that one trial doesn't enter twice the same group ==========
    np.random.shuffle(idx)

    new_idx = idx
    inds_micro_avg = []
    for itrial in range(n_epochs-10):
        is_not_ok = True
        while is_not_ok:
            inds_toappend = new_idx[:navg]
            if len(inds_toappend) == len(set(inds_toappend)):
                inds_micro_avg.append(inds_toappend)
                is_not_ok = False
                new_idx = new_idx[navg:]
            else:
                np.random.shuffle(new_idx)

    is_not_ok = True
    while is_not_ok:
        np.random.shuffle(new_idx)
        inds_final = []
        for itrial in range(10):
            inds_toappend = new_idx[itrial*navg:itrial*navg+navg]
            inds_final.append(inds_toappend)

        all_ok = True
        for k in range(10):
            if len(inds_final[k]) != len(set(inds_final[k])):
                all_ok = False

        if all_ok:
            is_not_ok = False

    inds_micro_avg.append(inds_final)
    inds_micro_avg = np.vstack(inds_micro_avg)

    micro_data = []
    for inds_avg in inds_micro_avg:
        print(inds_avg)
        average = np.mean(data[inds_avg],axis=0)
        micro_data.append(average)

    micro_data = np.asarray(micro_data)

    return micro_data

# ----------------------------------------------------------------------------------------------------------------------
def run_primitivepart_decoding_time_resolved(subject, which_primitives='11primitives', decim=None):
    """
    Function to decode the primitive types in the primitive part.
    :param subject: subject ID to load the epochs
    :param which_primitives: '11primitives' or 'rotVSsym'
    :param decim: set this to an integer if you want to further decimate the data
    """

    # 1 - select the either the pairs or the sequence data
    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject,filter='pairs')
    # we baseline the data and smooth it with a sliding window (size 25 dots i.e. 100 ms, every 4 ms, when the data is decimated by a factor of 4)
    epochs_for_decoding.apply_baseline()
    if decim is not None:
        print("--- we extra decimate the epochs ----")
        epochs_for_decoding.decimate(decim)
    print("---- the epochs are decimated by a factor of %i ----"%epochs_for_decoding._decim)

    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    # 2 - run either the 11 primitives decoder or the rotation / symmetry balanced decoder
    if which_primitives == '11primitives':
        epochs_for_decoding = epochs_for_decoding["first_or_second == 1 and violation == 0 and primitive != 'control'"]
        epochs_for_decoding.events[:, 2] = epochs_for_decoding.metadata['primitive_code'].values
        epochs_for_decoding.event_id = {'rotp1': 10, 'rotm1': 20, 'rotp2': 30, 'rotm2': 40, 'rotp3': 50, 'rotm3': 60,
                                        'sym_point': 70, 'H': 80, 'V': 90, 'A': 100, 'B': 110}

        epochs_for_decoding.equalize_event_counts(
            {'rotp1': 10, 'rotm1': 20, 'rotp2': 30, 'rotm2': 40, 'rotp3': 50, 'rotm3': 60,
             'sym_point': 70, 'H': 80, 'V': 90, 'A': 100, 'B': 110})

        suff = ''
        inds_train, inds_test = leave_one_block_out(epochs_for_decoding)
        scores, times = gat_classifier_categories(epochs_for_decoding, epochs_for_decoding.events[:, 2],
                                                                 inds_train, inds_test, micro_avg_or_not=False, scoring=None,
                                                                 predict_probability=False)

        results_decoding_path = '/11primitives/primitive_part/'
        scores_name = config.result_path + "/decoding/" + results_decoding_path + '/' + subject + suff
        results_dict = {'scores': scores, 'times': epochs_for_decoding.times}
        np.save(scores_name, results_dict)

    else:
        balanced_rot_sym_decoding(epochs_for_decoding, subject,
                                  splitting_function=leave_one_block_out,
                                  analysis_name='/rotationVSsymmetry/primitive_part/')


# ----------------------------------------------------------------------------------------------------------------------
def run_sequencepart_decoding_time_resolved(subject,decim=None):
    """
    Function to decode the 11 primitive types in the sequence part.
    """

    # 1 - select the sequence data
    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject,filter='sequence')
    # we baseline the data and smooth it with a sliding window (size 25 dots i.e. 100 ms, every 4 ms)
    epochs_for_decoding.apply_baseline()
    if decim is not None:
        epochs_for_decoding.decimate(decim)
    print("---- the epochs are decimated by a factor of %i ----"%epochs_for_decoding._decim)

    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    # 2 - run either the 11 primitives decoder or the rotation / symmetry balanced decoder
    epochs_for_decoding = epochs_for_decoding[
        "primitive_level1 != 'nan' and violation == 0 and primitive != 'control'"]
    epochs_for_decoding.events[:, 2] = epochs_for_decoding.metadata['primitive_code'].values
    epochs_for_decoding.event_id = {'rotp1': 10, 'rotm1': 20, 'rotp2': 30, 'rotm2': 40, 'rotp3': 50, 'rotm3': 60,
                                    'sym_point': 70, 'H': 80, 'V': 90, 'A': 100, 'B': 110}
    epochs_for_decoding.equalize_event_counts(
        {'rotp1': 10, 'rotm1': 20, 'rotp2': 30, 'rotm2': 40, 'rotp3': 50, 'rotm3': 60,
         'sym_point': 70, 'H': 80, 'V': 90, 'A': 100, 'B': 110})

    inds_train, inds_test = stratified_cv(epochs_for_decoding)

    scores, times = gat_classifier_categories(epochs_for_decoding, epochs_for_decoding.events[:, 2],
                                                             inds_train, inds_test, micro_avg_or_not=False, scoring=None,
                                                             predict_probability=False)

    results_decoding_path = '/11primitives/sequence_part/'
    scores_name = config.result_path + "/decoding/" + results_decoding_path + '/' + subject
    results_dict = {'scores': scores, 'times': epochs_for_decoding.times}
    np.save(scores_name, results_dict)

# ----------------------------------------------------------------------------------------------------------------------
def balanced_rot_sym_decoding(epochs, subject, splitting_function, analysis_name, micro_avg_or_not,
                              sliding_window_or_not, block_type='pairs', n_jobs=8):
    """
    Decoding when we balance the presented pairs on the screen in terms of it they belong to a rotation or a symmetry.
    :param epochs: epochs for
    :param subject:
    :param splitting_function:
    :param analysis_name:
    :param micro_avg_or_not:
    :param sliding_window_or_not: True, False or None if it is not needed
    :param full_window: Set it to true if you want to compute the decoding on the full time window specified by tmin and tmax
    :param tmin:
    :param tmax:
    :return:
    """

    epochs_balanced = balance_presented_pairs(epochs, block_type=block_type)
    if sliding_window_or_not:
        epochs_balanced = epoching_funcs.sliding_window(epochs_balanced,sliding_window_step=1)

    inds_train, inds_test = splitting_function(epochs_balanced, return_labels=False)
    labels = 1 * (epochs_balanced.metadata["rotation_or_symmetry"].values == "rotation")

    scores, times = gat_classifier_categories(epochs_balanced, labels, inds_train, inds_test, micro_avg_or_not,
                                              scoring="roc_auc",n_jobs=n_jobs)

    scores_name = config.result_path + "/decoding/" + analysis_name + '/' + subject
    results_dict = {'scores': scores, 'times': times}
    np.save(scores_name, results_dict)




# ----------------------------------------------------------------------------------------------------------------------
def leave_one_block_out(epochs_balanced, return_labels=False):
    """
    This function returns the indices for different folds corresponding to the different runs
    :param epochs_balanced: Input the epochs that have been already balanced in terms of rotation and symmetries
    :return: indices for cross validation
    """

    run_numb = np.unique(epochs_balanced.metadata["run_number"])
    labels = epochs_balanced.metadata['primitive_code'].values

    inds_train = []
    inds_test = []
    labels_train = []
    labels_test = []

    for run in run_numb:
        idx_train = np.where(epochs_balanced.metadata["run_number"].values != run)[0]
        idx_test = np.where(epochs_balanced.metadata["run_number"].values == run)[0]
        inds_train.append(idx_train)
        inds_test.append(idx_test)
        labels_train.append(labels[idx_train])
        labels_test.append(labels[idx_test])

    if return_labels:
        return inds_train, inds_test, labels_train, labels_test

    return inds_train, inds_test

# ----------------------------------------------------------------------------------------------------------------------
def stratified_cv(epochs_balanced, cv=4, return_labels=True):
    """
    This function returns the indices for different folds corresponding to the different runs
    :param epochs_balanced: Input the epochs that have been already balanced in terms of rotation and symmetries
    :return: indices for cross validation
    """
    skf = StratifiedKFold(n_splits=cv)
    primitive_codes = (epochs_balanced.metadata["primitive_code"]).values
    inds_train = []
    inds_test = []
    labels_train = []
    labels_test = []
    for train_index, test_index in skf.split(epochs_balanced.get_data(), primitive_codes):
        inds_train.append(train_index)
        inds_test.append(test_index)
        labels_train.append(primitive_codes[train_index])
        labels_test.append(primitive_codes[test_index])
    if return_labels:
        return inds_train, inds_test, labels_train, labels_test

    return inds_train, inds_test


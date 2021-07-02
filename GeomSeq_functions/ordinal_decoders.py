import glob
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from GeomSeq_analyses import config
from GeomSeq_functions import utils, epoching_funcs
from GeomSeq_functions.primitive_decoding_funcs import gat_classifier_categories

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack

# ______________________________________________________________________________________________________________________
def run_ordinal_decoding_GAT(subject):
    """
    Wrapper function that generates all the Generalization across time data that is plotted in figure 6, S3 - S6.
    """

    utils.create_folder(config.result_path + "/decoding/ordipos_GAT/")

    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)
    scores_dict = decode_position_in_component(epochs_for_decoding, control=False)
    for key in scores_dict.keys():
        scores_name = config.result_path + "/decoding/ordipos_GAT/" + subject + key
        np.save(scores_name, scores_dict[key])

    # =================== now run the control analysis ===========================
    scores_dict = decode_position_in_component(epochs_for_decoding, control=True)
    for key in scores_dict.keys():
        scores_name = config.result_path + "/decoding/ordipos_GAT/" + subject + key + '_control'
        np.save(scores_name, scores_dict[key])

# ______________________________________________________________________________________________________________________
def decode_position_in_component(epochs_for_decoding,control=False,micro_avg_or_not = False,predict_proba=True):
    """
    GAT matrices for the decoding of the ordinal positions in the following cases:
    'train4diag_test4seg': We train on 4diagonals and test on 4segments ,
    'train4seg_test4diag': Vice versa,
    'train2arcs_test2squares': We train on 2arcs and test on 2squares,
    'train2squares_test2arcs': Vice-versa,
    '2arcs_2squares': We train and test on the data of 2arcs and 2squares from 3 blocks and test on the 4th (CV),
    '4segments_4diagonals': Same idea but for 4segments and 4diagonals,
    'train2_test4': Train on 2arcs and 2squares, test on 4segments and 4diagonals,
    'train4_test2': Vice-versa,

    :param epochs_for_decoding:
    :param micro_avg_or_not:
    :param predict_proba:
    :param control: Set it to True if you want to run the control analysis (with irregular and repeat)
    :return:
    """

    # ============================= define the epochs for ordinal information analysis =================================
    epochs_4diagonals = epochs_for_decoding["sequence == '4diagonals' and violation ==0"]
    epochs_4segments = epochs_for_decoding["sequence == '4segments' and violation ==0"]
    epochs_2arcs = epochs_for_decoding["sequence == '2arcs' and violation ==0"]
    epochs_2squares = epochs_for_decoding["sequence == '2squares' and violation ==0"]

    if control:
        print("We are running the control analyses.")
        # We replace the data by the data coming from the 'control' sequences, for which there is no component structure.
        # The metadata stays the same.
        epochs_4diagonals._data = epochs_for_decoding["sequence == 'repeat' and violation ==0"]._data
        epochs_4segments._data = epochs_for_decoding["sequence == 'irregular' and violation ==0"]._data
        epochs_2arcs._data = epochs_for_decoding["sequence == 'repeat' and violation ==0"]._data
        epochs_2squares._data = epochs_for_decoding["sequence == 'irregular' and violation ==0"]._data

    # ============================= CONCATENATE THE EPOCHS ==================================
    epochs_4DS = mne.concatenate_epochs([epochs_4diagonals, epochs_4segments])
    epochs_2AS = mne.concatenate_epochs([epochs_2arcs, epochs_2squares])
    epochs_all = mne.concatenate_epochs([epochs_2AS,epochs_4DS])
    # ============================= LABELS ==================================================
    labels_4DS = np.asarray([int(k) for k in epochs_4DS.metadata["WithinComponentPosition"].values])
    labels_2AS = np.asarray([int(k) for k in epochs_2AS.metadata["WithinComponentPosition"].values])
    labels_all = np.asarray([int(k) for k in epochs_all.metadata["WithinComponentPosition"].values])
    # ============================= INDICES =================================================
    inds_4DS = np.asarray(['D'] * len(epochs_4diagonals) + ['S'] * len(epochs_4segments))
    inds_2AS = np.asarray(['A'] * len(epochs_2arcs) + ['S'] * len(epochs_2squares))
    inds_all = np.asarray(['2']* len(epochs_2AS) +['4']* len(epochs_4DS))

    # ============================= NOW DECODE ============================================
    scores_train_4D_test_4S, times = gat_classifier_categories(epochs_4DS, labels_4DS,
                                                                       [np.where(inds_4DS == 'D')[0]],
                                                                       [np.where(inds_4DS == 'S')[0]], micro_avg_or_not,predict_probability=predict_proba)
    scores_train_4S_test_4D, times = gat_classifier_categories(epochs_4DS, labels_4DS,
                                                                       [np.where(inds_4DS == 'S')[0]],
                                                                       [np.where(inds_4DS == 'D')[0]], micro_avg_or_not,predict_probability=predict_proba)
    scores_train_2A_test_2S, times = gat_classifier_categories(epochs_2AS, labels_2AS,
                                                                       [np.where(inds_2AS == 'A')[0]],
                                                                       [np.where(inds_2AS == 'S')[0]], micro_avg_or_not,predict_probability=predict_proba)
    scores_train_2S_test_2A, times = gat_classifier_categories(epochs_2AS, labels_2AS,
                                                                       [np.where(inds_2AS == 'S')[0]],
                                                                       [np.where(inds_2AS == 'A')[0]], micro_avg_or_not,predict_probability=predict_proba)
    scores_24, times = gat_classifier_categories(epochs_all, labels_all,
                                                                       [np.where(inds_all == '2')[0]],
                                                                       [np.where(inds_all == '4')[0]], micro_avg_or_not,predict_probability=predict_proba)
    scores_42, times = gat_classifier_categories(epochs_all, labels_all,
                                                                       [np.where(inds_all == '4')[0]],
                                                                       [np.where(inds_all == '2')[0]], micro_avg_or_not,predict_probability=predict_proba)

    # ====== when pulling together the 2arcs and 2squares sequences (and 4segments and 4diagonal sequences), we cross validate across runs ======
    inds_train_4 = []
    inds_test_4 = []
    inds_train_2 = []
    inds_test_2 = []

    run_values = np.unique(epochs_4DS.metadata["run_number"].values)

    for run in run_values:
        print(run)
        inds_train_4.append(np.where(epochs_4DS.metadata["run_number"]!=run)[0])
        inds_test_4.append(np.where(epochs_4DS.metadata["run_number"]==run)[0])
        inds_train_2.append(np.where(epochs_2AS.metadata["run_number"]!=run)[0])
        inds_test_2.append(np.where(epochs_2AS.metadata["run_number"]==run)[0])

    scores_2, times = gat_classifier_categories(epochs_2AS, labels_2AS, inds_train_2,inds_test_2, micro_avg_or_not,predict_probability=predict_proba)
    scores_4, times = gat_classifier_categories(epochs_4DS, labels_4DS,inds_train_4,inds_test_4, micro_avg_or_not,predict_probability=predict_proba)

    scores_dict = {'train4diag_test4seg': scores_train_4D_test_4S,
                   'train4seg_test4diag': scores_train_4S_test_4D,
                   'train2arcs_test2squares': scores_train_2A_test_2S,
                   'train2squares_test2arcs': scores_train_2S_test_2A,
                   '2arcs_2squares':scores_2,
                   '4segments_4diagonals': scores_4,
                   'train2_test4':scores_24,
                   'train4_test2':scores_42,
                   'times': epochs_for_decoding.times}

    return scores_dict

# ______________________________________________________________________________________________________________________
def train_decoder_window(epochs,labels,tmin,tmax,scoring=None,predict_probability=True):
    """
    Small function to crop the epochs and train an SVM decoder on the cropped epochs with the corresponding labels
    """
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = epochs.get_data()
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability = predict_probability))
    time_gen = GeneralizingEstimator(clf, scoring=scoring, n_jobs=8, verbose=True)
    time_gen.fit(X,labels)

    return time_gen


# ______________________________________________________________________________________________________________________
def decode_ordinal_position_oneSequence(subject, tmin=0.3, tmax=0.5,control=False,return_trained_decoders=False):
    """
    Decoding the ordinal position when training on a given window [tmin,tmax], and testing on the full sequence.
    This function outputs a dictionnary that contains the results when
    'train4diag_test4seg' : training on 4diagonals and testing on 4segments
    'train4seg_test4diag' : training on 4segments and testing on 4diagonals
    'train2squares_test2arcs' : training on 2squares testing on 2arcs
    'train2arcs_test2squares' : training on 2arcs testing on 2squares

    :param subject:
    :param baseline_or_not: Baselining the epochs, same parameter for the training and testing epochs (while the training epochs
    are on items and the testing ones are on the full sequence)
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin: minimal time for training times of the decoder
    :param tmax: max time for training times
    :param control: set it to True if you want to run the analysis on irregular and repeat sequences
    :return:
    """

    # __________________________________________________________________________________________________
    # ==== load the 1item epochs ====
    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    # ==== select the epochs for the different training sequences ====
    epo_4seg = epochs_for_decoding["sequence == '4segments'"]
    epo_4diag = epochs_for_decoding["sequence == '4diagonals'"]
    epo_2arcs = epochs_for_decoding["sequence == '2arcs'"]
    epo_2squares = epochs_for_decoding["sequence == '2squares'"]

    if control:
        # we replace the data (and not the metadata) by the data coming from the 'control' sequences, for which there is no component structure.
        epo_4seg._data = epochs_for_decoding["sequence == 'irregular'"]._data
        epo_4diag._data = epochs_for_decoding["sequence == 'repeat'"]._data
        epo_2arcs._data = epochs_for_decoding["sequence == 'repeat'"]._data
        epo_2squares._data = epochs_for_decoding["sequence == 'irregular'"]._data

    # ==== the training labels are the position in the the component ====
    label_4seg = np.asarray([int(k) for k in epo_4seg.metadata["WithinComponentPosition"].values])
    label_4diag = np.asarray([int(k) for k in epo_4diag.metadata["WithinComponentPosition"].values])
    label_2arcs = np.asarray([int(k) for k in epo_2arcs.metadata["WithinComponentPosition"].values])
    label_2squares = np.asarray([int(k) for k in epo_2squares.metadata["WithinComponentPosition"].values])
    # ====== training the decoders =======
    dec_4seg = train_decoder_window(epo_4seg, label_4seg, tmin, tmax)
    dec_4diag = train_decoder_window(epo_4diag, label_4diag, tmin, tmax)
    dec_2arcs = train_decoder_window(epo_2arcs, label_2arcs, tmin, tmax)
    dec_2squares = train_decoder_window(epo_2squares, label_2squares, tmin, tmax)

    if return_trained_decoders:
        return dec_4seg, dec_4diag, dec_2arcs, dec_2squares

    # __________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO A SEQUENCE   ==================

    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_sequence')

    epo_4seg_full = epochs_for_decoding_full["sequence == '4segments'"]
    epo_4diag_full = epochs_for_decoding_full["sequence == '4diagonals'"]
    epo_2arcs_full = epochs_for_decoding_full["sequence == '2arcs'"]
    epo_2squares_full = epochs_for_decoding_full["sequence == '2squares'"]

    if control:
        epo_4seg_full._data = epochs_for_decoding_full["sequence == 'irregular'"]._data
        epo_4diag_full._data = epochs_for_decoding_full["sequence == 'repeat'"]._data
        epo_2arcs_full._data = epochs_for_decoding_full["sequence == 'repeat'"]._data
        epo_2squares_full._data = epochs_for_decoding_full["sequence == 'irregular'"]._data

    y_train4diag_test4seg = dec_4diag.predict_proba(epo_4seg_full.get_data())
    y_train4seg_test4diag = dec_4seg.predict_proba(epo_4diag_full.get_data())
    y_train2squares_test2arcs = dec_2squares.predict_proba(epo_2arcs_full.get_data())
    y_train2arcs_test2squares = dec_2arcs.predict_proba(epo_2squares_full.get_data())

    dist_train4diag_test4seg = dec_4diag.decision_function(epo_4seg_full.get_data())
    dist_train4seg_test4diag = dec_4seg.decision_function(epo_4diag_full.get_data())
    dist_train2squares_test2arcs = dec_2squares.decision_function(epo_2arcs_full.get_data())
    dist_train2arcs_test2squares = dec_2arcs.decision_function(epo_2squares_full.get_data())

    results_train4diag_test4seg = {'y_preds': np.asarray(y_train4diag_test4seg),
                                                  'times': epochs_for_decoding_full.times,'distances':dist_train4diag_test4seg}
    results_train4seg_test4diag = {'y_preds': np.asarray(y_train4seg_test4diag),
                                                  'times': epochs_for_decoding_full.times,'distances':dist_train4seg_test4diag}
    results_train2squares_test2arcs = {'y_preds': np.asarray(y_train2squares_test2arcs),
                                                  'times': epochs_for_decoding_full.times,'distances':dist_train2squares_test2arcs}
    results_train2arcs_test2squares = {'y_preds': np.asarray(y_train2arcs_test2squares),
                                                  'times': epochs_for_decoding_full.times,'distances':dist_train2arcs_test2squares}
    suffix_control=''
    if control:
        suffix_control='_control'

    save_path = config.result_path + '/decoding/ordinal_code_full_seq/'
    utils.create_folder(save_path)
    np.save(save_path + subject + '_train4diag_test4seg'+suffix_control+'.npy', results_train4diag_test4seg)
    np.save(save_path + subject + '_train4seg_test4diag'+suffix_control+'.npy', results_train4seg_test4diag)
    np.save(save_path + subject + '_train2squares_test2arcs'+suffix_control+'.npy', results_train2squares_test2arcs)
    np.save(save_path + subject + '_train2arcs_test2squares'+suffix_control+'.npy', results_train2arcs_test2squares)



# ______________________________________________________________________________________________________________________
def decode_ordinal_position_oneSequence_CV(subject,tmin=0.3, tmax=0.5,control=False):
    """
    Decoding the ordinal position when training on (4segments and 4diagonals) or (2arcs and 2squares) on a given window [tmin,tmax]
    for run 'n', and testing on the full sequence of (4segments and 4diagonals) or (2arcs and 2squares)
    for all the runs that are not 'n'. And we cross-validate 4 times across the 4 runs.

    :param subject:
    :param baseline_or_not: Baselining the epochs, same parameter for the training and testing epochs (while the training epochs
    are on items and the testing ones are on the full sequence)
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin: minimal time for training times of the decoder
    :param tmax: max time for training times
    :param control: set it to True if you want to run the analysis on irregular and repeat sequences
    :return:
    """

    # __________________________________________________________________________________________________
    # ================================= training data  =================================================

    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    # ==== select the epochs for the different training sequences ====
    epo_4SD = mne.concatenate_epochs([epochs_for_decoding["sequence == '4segments'"],epochs_for_decoding["sequence == '4diagonals'"]])
    epo_2AS = mne.concatenate_epochs([epochs_for_decoding["sequence == '2arcs'"],epochs_for_decoding["sequence == '2squares'"]])

    if control:
        epo_control = mne.concatenate_epochs([epochs_for_decoding["sequence == 'irregular'"], epochs_for_decoding["sequence == 'repeat'"]])
        epo_4SD._data = epo_control._data
        epo_2AS._data = epo_control._data

    # ==== the training labels are the position in the the component ====
    label_4SD = np.asarray([int(k) for k in epo_4SD.metadata["WithinComponentPosition"].values])
    label_2AS = np.asarray([int(k) for k in epo_2AS.metadata["WithinComponentPosition"].values])
    # __________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO A SEQUENCE   ==================
    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_sequence')

    epo_full_4SD = mne.concatenate_epochs([epochs_for_decoding_full["sequence == '4segments'"],epochs_for_decoding_full["sequence == '4diagonals'"]])
    epo_full_2AS = mne.concatenate_epochs([epochs_for_decoding_full["sequence == '2arcs'"],epochs_for_decoding_full["sequence == '2squares'"]])

    if control:
        epo_full_4SD = mne.concatenate_epochs(
            [epochs_for_decoding_full["sequence == 'irregular'"], epochs_for_decoding_full["sequence == 'repeat'"]])
        epo_full_2AS = mne.concatenate_epochs(
            [epochs_for_decoding_full["sequence == 'irregular'"], epochs_for_decoding_full["sequence == 'repeat'"]])

    y_pred_4SD = []
    y_pred_2AS = []
    distances_4SD = []
    distances_2AS = []

    for run in range(2,6):
        print(run)
        inds_train_4SD = np.where(epo_4SD.metadata["run_number"]!=run)[0]
        dec_4SD = train_decoder_window(epo_4SD[inds_train_4SD], label_4SD[inds_train_4SD], tmin, tmax)
        distances_4SD.append(dec_4SD.decision_function(epo_full_4SD["run_number == %i"%run].get_data()))
        y_pred_4SD.append(dec_4SD.predict_proba(epo_full_4SD["run_number == %i"%run].get_data()))
        # _______________________________________________________________________________________
        inds_train_2AS = np.where(epo_2AS.metadata["run_number"]!=run)[0]
        dec_2AS= train_decoder_window(epo_2AS[inds_train_2AS], label_2AS[inds_train_2AS], tmin, tmax)
        distances_2AS.append(dec_2AS.decision_function(epo_full_2AS["run_number == %i"%run].get_data()))
        y_pred_2AS.append(dec_2AS.predict_proba(epo_full_2AS["run_number == %i"%run].get_data()))

    results_4SD = {'y_preds': np.asarray(y_pred_4SD),'times': epo_full_4SD.times,'distances':np.asarray(distances_4SD)}
    results_2AS = {'y_preds': np.asarray(y_pred_2AS),'times': epo_full_2AS.times,'distances':np.asarray(distances_2AS)}

    suffix_control=''
    if control:
        suffix_control='_control'
    save_path = config.result_path + '/decoding/ordinal_code_full_seq/'
    np.save(save_path + subject +  '_4segments_4diagonals'+suffix_control+'.npy', results_4SD)
    np.save(save_path + subject +  '_2arcs_2squares'+suffix_control+'.npy', results_2AS)


# ______________________________________________________________________________________________________________________
def decode_ordinal_position_oneSequence_train24_test42(subject, tmin=0.3, tmax=0.5):
    """
    Decoding the ordinal position when training on a given window [tmin,tmax], and testing on the full sequence.
    Here we train on the data of 2arcs and 2squares and test on 4diagonals and 4segments 'train2_test4'
    We do the opposite also : 'train4_test2'
    :param subject:
    :param baseline_or_not:
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin:
    :param tmax:
    :return:
    """

    # __________________________________________________________________________________________________
    # ==================== train on the averaged data  =================================================
    # ==== load the 1item epochs ====

    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)
    # ==== select the epochs for the different training sequences ====

    epo_4SD = mne.concatenate_epochs([epochs_for_decoding["sequence == '4segments'"],epochs_for_decoding["sequence == '4diagonals'"]])
    epo_2AS = mne.concatenate_epochs([epochs_for_decoding["sequence == '2arcs'"],epochs_for_decoding["sequence == '2squares'"]])

    # ==== the training labels are the position in the the component ====
    label_4 = np.asarray([int(k) for k in epo_4SD.metadata["WithinComponentPosition"].values])
    label_2 = np.asarray([int(k) for k in epo_2AS.metadata["WithinComponentPosition"].values])

    # ====== training the decoders =======
    dec_4 = train_decoder_window(epo_4SD, label_4, tmin, tmax)
    dec_2 = train_decoder_window(epo_2AS, label_2, tmin, tmax)
    # __________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO A SEQUENCE   ==================

    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_sequence')

    epo_4SD_full = mne.concatenate_epochs(
        [epochs_for_decoding_full["sequence == '4segments'"], epochs_for_decoding_full["sequence == '4diagonals'"]])
    epo_2AS_full = mne.concatenate_epochs(
        [epochs_for_decoding_full["sequence == '2arcs'"], epochs_for_decoding_full["sequence == '2squares'"]])

    y_train4_test2 = dec_4.predict_proba(epo_2AS_full.get_data())
    y_train2_test4 = dec_2.predict_proba(epo_4SD_full.get_data())

    dist_train4_test2 = dec_4.decision_function(epo_2AS_full.get_data())
    dist_train2_test4 = dec_2.decision_function(epo_4SD_full.get_data())

    results_train4_test2= {'y_preds': np.asarray(y_train4_test2),
                                                  'times': epochs_for_decoding_full.times,'distances':dist_train4_test2}
    results_train2_test4 = {'y_preds': np.asarray(y_train2_test4),
                                                  'times': epochs_for_decoding_full.times,'distances':dist_train2_test4}

    save_path = config.result_path + '/decoding/ordinal_code_full_seq/'
    utils.create_folder(save_path)
    np.save(save_path + subject + '_train4_test2.npy', results_train4_test2)
    np.save(save_path + subject + '_train2_test4.npy', results_train2_test4)


# ______________________________________________________________________________________________________________________
def decode_ordinal_position_allBlocks(subject,control=False):
    """
    This function trains the component number decoder on the data from one sequence type and tests it on the full mini-blocks devoted to the sequence.
    The testing set is n_runs X 12 (number of repetitions of a given sequence within a run)

    :param subject:
    :param baseline_or_not: This tells if we have baselined the data for training
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin: The time window on which we want to train the decoder
    :param tmax: The time window on which we want to train the decoder
    :return:
    """


    # ===== get the trained decoders =====

    dec_4seg, dec_4diag, dec_2arcs, dec_2squares = decode_ordinal_position_oneSequence(subject, tmin=0.3, tmax=0.5, control=control, return_trained_decoders=True)
    # ___________________________________________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    # ___________________________________________________________________________________________________________________________________

    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_block')

    epo_full_4seg = epochs_for_decoding_full["sequence == '4segments'"]
    epo_full_4diag = epochs_for_decoding_full["sequence == '4diagonals'"]
    epo_full_2arcs = epochs_for_decoding_full["sequence == '2arcs'"]
    epo_full_2squ = epochs_for_decoding_full["sequence == '2squares'"]

    if control:
        epo_full_4seg._data = epochs_for_decoding_full["sequence == 'irregular'"]._data
        epo_full_4diag._data = epochs_for_decoding_full["sequence == 'repeat'"]._data
        epo_full_2arcs._data = epochs_for_decoding_full["sequence == 'repeat'"]._data
        epo_full_2squ._data = epochs_for_decoding_full["sequence == 'irregular'"]._data

    y_train4diag_test4seg = dec_4diag.predict_proba(epo_full_4seg)
    y_train4seg_test4diag = dec_4seg.predict_proba(epo_full_4diag)
    y_train2squares_test2arcs = dec_2squares.predict_proba(epo_full_2arcs)
    y_train2arcs_test2squares = dec_2arcs.predict_proba(epo_full_2squ)

    dist_train4diag_test4seg = dec_4diag.decision_function(epo_full_4seg)
    dist_train4seg_test4diag = dec_4seg.decision_function(epo_full_4diag)
    dist_train2squares_test2arcs = dec_2squares.decision_function(epo_full_2arcs)
    dist_train2arcs_test2squares = dec_2arcs.decision_function(epo_full_2squ)

    # ============== back to miniblock reorganizes the data into 4 (runs) X time series of the 12 repetitions ==================================

    times = epochs_for_decoding_full.times

    results_train4diag_test4seg = {'y_preds': np.asarray(y_train4diag_test4seg),
                                   'distances':dist_train4diag_test4seg,'times':times}
    results_train4seg_test4diag = {'y_preds': np.asarray(y_train4seg_test4diag),
                                    'distances':dist_train4seg_test4diag,'times':times}
    results_train2squares_test2arcs = {'y_preds': np.asarray(y_train2squares_test2arcs),
                                        'distances':dist_train2squares_test2arcs,'times':times}
    results_train2arcs_test2squares = {'y_preds': np.asarray(y_train2arcs_test2squares),
                                       'distances':dist_train2arcs_test2squares,'times':times}

    suffix_control=''
    if control:
        suffix_control='_control'
    save_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/'
    utils.create_folder(save_path)
    np.save(save_path + subject +  '_train4diag_test4seg'+suffix_control+'.npy', results_train4diag_test4seg)
    np.save(save_path + subject + '_train4seg_test4diag'+suffix_control+'.npy', results_train4seg_test4diag)
    np.save(save_path + subject +  '_train2squares_test2arcs'+suffix_control+'.npy', results_train2squares_test2arcs)
    np.save(save_path + subject + '_train2arcs_test2squares'+suffix_control+'.npy', results_train2arcs_test2squares)


# ______________________________________________________________________________________________________________________
def decode_ordinal_position_allBlocks_CV(subject, tmin=0.3, tmax=0.5,control=False):
    """
    This function trains the ordinal number decoder on the data from one sequence type and tests it on the full mini-blocks
    The testing set is n_runs X 12 (number of repetitions of a given sequence within a run)
    For example, we train on 3 out of 4 blocks for 4segments and 4 digonals and test on the 4th. We repeat this procedure 4 times.
    (This is CV across blocks)
    :param subject:
    :param baseline_or_not: This tells if we have baselined the data for training
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin: The time window on which we want to train the decoder
    :param tmax: The time window on which we want to train the decoder
    :return:
    """
    # __________________________________________________________________________________________________
    # ==================== training epochs =================================================
    # ==== load the 1item epochs ====

    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    epo_4SD = mne.concatenate_epochs([epochs_for_decoding["sequence == '4segments'"],epochs_for_decoding["sequence == '4diagonals'"]])
    epo_2AS = mne.concatenate_epochs([epochs_for_decoding["sequence == '2arcs'"],epochs_for_decoding["sequence == '2squares'"]])

    if control:
        epo_for_data = mne.concatenate_epochs(
            [epochs_for_decoding["sequence == 'irregular'"], epochs_for_decoding["sequence == 'repeat'"]])
        epo_4SD._data = epo_for_data._data
        epo_2AS._data = epo_for_data._data

    # ==== the training labels are the ordinal position in the component ====
    label_4SD = np.asarray([int(k) for k in epo_4SD.metadata["WithinComponentPosition"].values])
    label_2AS = np.asarray([int(k) for k in epo_2AS.metadata["WithinComponentPosition"].values])

    # ___________________________________________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    # ___________________________________________________________________________________________________________________________________

    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_block')

    y_pred_4SD = []
    y_pred_2AS = []
    distances_4SD = []
    distances_2AS = []

    run_numb = np.unique(epochs_for_decoding_full.metadata["run_number"].values)
    for run in run_numb:

        # % % %  4diagonals and 4 segments % % %
        # ======== training on all the blocks and all the items appart from the ones belonging to run number "run"
        inds_train_4SD = np.where(epo_4SD.metadata["run_number"]!=run)[0]
        dec_4SD = train_decoder_window(epo_4SD[inds_train_4SD], label_4SD[inds_train_4SD], tmin, tmax)
        # ======== re-concatenate the data corresponding to the mini-block 'run' ======
        epo_4seg_data = epochs_for_decoding_full["sequence == '4segments' and run_number == %i"%run]
        epo_4diag_data = epochs_for_decoding_full["sequence == '4diagonals' and run_number == %i"%run]

        if control:
            epo_4seg_data = epochs_for_decoding_full["sequence == 'irregular' and run_number == %i" % run]
            epo_4diag_data = epochs_for_decoding_full["sequence == 'repeat' and run_number == %i" % run]

        epo_full_4SD_data = np.vstack(np.asarray([epo_4seg_data,epo_4diag_data]))
        distances_4SD.append(dec_4SD.decision_function(epo_full_4SD_data))
        y_pred_4SD.append(dec_4SD.predict_proba(epo_full_4SD_data))

        # % % %  2arcs and 2crosses % % %
        inds_train_2AS = np.where(epo_2AS.metadata["run_number"]!=run)[0]
        dec_2AS= train_decoder_window(epo_2AS[inds_train_2AS], label_2AS[inds_train_2AS], tmin, tmax)

        epo_2arc_data = epochs_for_decoding_full["sequence == '2arcs' and run_number == %i"%run]
        epo_2squ_data = epochs_for_decoding_full["sequence == '2squares' and run_number == %i"%run]

        if control:
            epo_2arc_data = epochs_for_decoding_full["sequence == 'repeat' and run_number == %i" % run]
            epo_2squ_data = epochs_for_decoding_full["sequence == 'irregular' and run_number == %i" % run]

        epo_full_2AS_data = np.vstack(np.asarray([epo_2arc_data,epo_2squ_data]))
        distances_2AS.append(dec_2AS.decision_function(epo_full_2AS_data))
        y_pred_2AS.append(dec_2AS.predict_proba(epo_full_2AS_data))

    times = epochs_for_decoding_full.times

    results_4SD = {'y_preds': np.asarray(y_pred_4SD),
                                   'distances':distances_4SD,'times':times}
    results_2AS = {'y_preds': np.asarray(y_pred_2AS),
                                    'distances':distances_2AS,'times':times}
    suffix_control = ''
    if control:
        suffix_control='_control'

    save_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/'
    np.save(save_path + subject + '_4segments_4diagonals'+suffix_control+'.npy', results_4SD)
    np.save(save_path + subject + '_2arcs_2squares'+suffix_control+'.npy', results_2AS)



# ______________________________________________________________________________________________________________________
def decode_ordinal_position_allBlocks_train42_test24(subject,tmin=0.2, tmax=0.6):
    """
    This function trains the component number decoder on the data from one sequence type and tests it on the full mini-blocks devoted to the sequence.
    The testing set is n_runs X 12 (number of repetitions of a given sequence within a run)

    :param subject:
    :param baseline_or_not: This tells if we have baselined the data for training
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin: The time window on which we want to train the decoder
    :param tmax: The time window on which we want to train the decoder
    :return:
    """
    # __________________________________________________________________________________________________
    # ==================== train on the averaged data  =================================================
    # ==== load the 1item epochs ====
    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    # ==== select the epochs for the different training sequences ====

    epo_4SD = mne.concatenate_epochs([epochs_for_decoding["sequence == '4segments'"],epochs_for_decoding["sequence == '4diagonals'"]])
    epo_2AS = mne.concatenate_epochs([epochs_for_decoding["sequence == '2arcs'"],epochs_for_decoding["sequence == '2squares'"]])

    # ==== the training labels are the position in the the component ====
    label_4 = np.asarray([int(k) for k in epo_4SD.metadata["WithinComponentPosition"].values])
    label_2 = np.asarray([int(k) for k in epo_2AS.metadata["WithinComponentPosition"].values])
    # ====== training the decoders =======
    dec_4 = train_decoder_window(epo_4SD, label_4, tmin, tmax)
    dec_2 = train_decoder_window(epo_2AS, label_2, tmin, tmax)

    # ___________________________________________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    # ___________________________________________________________________________________________________________________________________
    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_block')

    epo_4SD_full = mne.concatenate_epochs(
        [epochs_for_decoding_full["sequence == '4segments'"], epochs_for_decoding_full["sequence == '4diagonals'"]])
    epo_2AS_full = mne.concatenate_epochs(
        [epochs_for_decoding_full["sequence == '2arcs'"], epochs_for_decoding_full["sequence == '2squares'"]])

    y_train4_test2 = dec_4.predict_proba(epo_2AS_full)
    y_train2_test4 = dec_2.predict_proba(epo_4SD_full)
    dist_train4_test2 = dec_4.decision_function(epo_2AS_full)
    dist_train2_test4 = dec_2.decision_function(epo_4SD_full)

    times = epochs_for_decoding.times
    results_train4_test2= {'y_preds': np.asarray(y_train4_test2),
                                   'distances':dist_train4_test2,'times':times}
    results_train2_test4 = {'y_preds': np.asarray(y_train2_test4),
                                    'distances':dist_train2_test4,'times':times}

    save_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/'
    np.save(save_path + subject + '_train4_test2.npy', results_train4_test2)
    np.save(save_path + subject + '_train2_test4.npy', results_train2_test4)


# ______________________________________________________________________________________________________________________
def decode_ordinal_position_allBlocks_repeat_irregular(subject, tmin=0.2, tmax=0.6):
    """
    This function trains the component number decoder on the data from one sequence type and tests it on the full mini-blocks devoted to the sequence.
    The testing set is n_runs X 12 (number of repetitions of a given sequence within a run)


    :param subject:
    :param baseline_or_not: This tells if we have baselined the data for training
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param tmin: The time window on which we want to train the decoder
    :param tmax: The time window on which we want to train the decoder
    :return:
    """

    # __________________________________________________________________________________________________
    # ==================== training epochs =================================================
    # ==== load the 1item epochs ====
    epochs_for_decoding = epoching_funcs.load_and_concatenate_epochs(subject, filter='sequence')
    epochs_for_decoding.decimate(4)
    epochs_for_decoding = epoching_funcs.sliding_window(epochs_for_decoding)

    epo_2arcs = epochs_for_decoding["sequence == '2arcs'"]
    epo_irregular = epochs_for_decoding["sequence == 'irregular'"]
    epo_repeat = epochs_for_decoding["sequence == 'repeat'"]

    # ==== the training labels are the position in the the component ====
    label_irregular = np.asarray([int(k) for k in epo_2arcs.metadata["WithinComponentPosition"].values])
    label_repeat = np.asarray([int(k) for k in epo_2arcs.metadata["WithinComponentPosition"].values])

    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    epochs_for_decoding_full = epoching_funcs.load_and_concatenate_epochs(subject, filter='full_block')

    distances_repeat = []
    y_pred_repeat = []
    distances_irregular = []
    y_pred_irregular = []

    run_numb = np.unique(epochs_for_decoding_full.metadata["run_number"].values)

    for run in run_numb:
        # ======== training on all the blocks and all the items appart from the ones belonging to run number "run"
        inds_train_irregular = np.where(epo_irregular.metadata["run_number"]!=run)[0]
        dec_irregular = train_decoder_window(epo_irregular[inds_train_irregular], label_irregular[inds_train_irregular], tmin, tmax)
        epo_irregular_full = epochs_for_decoding_full["sequence == 'irregular' and run_number == %i"%run]
        distances_irregular.append(dec_irregular.decision_function(epo_irregular_full))
        y_pred_irregular.append(dec_irregular.predict_proba(epo_irregular_full))

        # _______________________________________________________________________________________
        inds_train_repeat = np.where(epo_repeat.metadata["run_number"]!=run)[0]
        dec_repeat = train_decoder_window(epo_repeat[inds_train_repeat], label_repeat[inds_train_repeat], tmin, tmax)
        epo_repeat_full = epochs_for_decoding_full["sequence == 'repeat' and run_number == %i"%run]
        distances_repeat.append(dec_repeat.decision_function(epo_repeat_full))
        y_pred_repeat.append(dec_repeat.predict_proba(epo_repeat_full))

    # ============== back to miniblock reorganizes the data into 4 (runs) X time series of the 12 repetitions ==================================

    times = epochs_for_decoding_full.times
    results_irregular = {'y_preds': np.asarray(y_pred_irregular),
                                   'distances':distances_irregular,'times':times}
    results_repeat = {'y_preds': np.asarray(y_pred_repeat),
                                    'distances':distances_repeat,'times':times}

    save_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/'
    np.save(save_path + subject + '_irregular.npy', results_irregular)
    np.save(save_path + subject + '_repeat.npy', results_repeat)

# ======================================================================================================================
# ================================== FOURIER TRANSFORM FUNCTIONS =======================================================
# ======================================================================================================================
# ______________________________________________________________________________________________________________________
def analysis_names(control=True):

    analyses = [['2arcs_2squares'], ['4segments_4diagonals'], ['train2arcs_test2squares', 'train2squares_test2arcs'],
                ['train4diag_test4seg', 'train4seg_test4diag'], ['train4_test2'], ['train2_test4'], ['repeat'],
                ['irregular']]
    list_saving_names = ['traintest_2arcs_2squares', 'traintest_4segments_4diagonals', 'train4_test4_average',
                         'train2_test2_average', 'train4_test2', 'train2_test4', 'repeat', 'irregular']

    if control:
        analyses = [['2arcs_2squares'], ['4segments_4diagonals'],
                    ['train2arcs_test2squares', 'train2squares_test2arcs'],
                    ['train4diag_test4seg', 'train4seg_test4diag']]
        list_saving_names = ['traintest_2arcs_2squares', 'traintest_4segments_4diagonals', 'train4_test4_average',
                             'train2_test2_average']

    return analyses, list_saving_names
# ______________________________________________________________________________________________________________________
def oscillations_ordinal_code():
    """
    This function computes the fourier transform of the projection on the decision vector for the ordinal code results.
    We average over the training time window (300-500 ms)
    """
    save_results_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/'
    save_fig_path = config.figure_path + '/decoding/decode_ordinal_position_fullblock/'
    utils.create_folder(save_fig_path)

    for control in [False,True]:
        analyses, list_saving_names = analysis_names(control=control)
        suffix = ''
        if control:
            suffix = '_control'
        for ii, anal_list in enumerate(analyses):
            for anal in anal_list:
                fig, signals, times = average_and_plot_ord_pos(analysis_list=[anal],control=control)
                plt.close('all')
                n_subj, n_times, n_ord_positions = signals.shape
                for ord_pos in range(n_ord_positions):
                    fft_results = compute_fft(signals[:,:,ord_pos],times)
                    save_path_cc = save_results_path+list_saving_names[ii]+'fft_results'+str(ord_pos+1)+suffix+'.npy'
                    np.save(save_path_cc,fft_results)

# ______________________________________________________________________________________________________________________
def average_and_plot_ord_pos(analysis_list,control=False):
    """
    This function averages the predictions across decoder's training times and epochs.
    The average of the predictions will be computed over the window of 300-500ms
    :param analysis_list: Here give the list of the analyses names that will be loaded and for which the results are going
    to be averaged afterwards.
    :return: figure handle, mean across training times and epochs, times
    """

    # ---- parameters for the plots -----
    figsize = [3.2, 1.6]
    labelsize = 6
    fontsize = 6
    linewidth = 0.7
    linewidth_zero = 1
    linewidth_other = 0.5
    ylim = {'distance': {2: [-0.12, 0.12], 4: [1.4, 1.62]}}

    # ----- list of the analyses that were run with cross-validation. We need to average for them also across CV folds -
    analyses_with_CV = ['4segments_4diagonals', '2arcs_2squares', 'repeat', 'irregular']

    # load the predicted distances : projection on the decision vector obtained from the decoding of the ordinal code
    predictions = []
    for k in range(len(analysis_list)):
        predictions_to_append, times = load_predicted_ordinal_positions(analysis=analysis_list[k], control=control)
        predictions.append(predictions_to_append)
    # we average across the analyses mentionned in the list
    predictions = np.mean(predictions, axis=0)

    if analysis_list[0] in analyses_with_CV:
        #  cross validation was computed across blocks. We average across blocks
        predictions = np.mean(predictions, axis=1)

    n_subj, n_epochs, n_train_times, n_test_times, n_cat = predictions.shape

    # ======== average across times and epochs to obtain the final time series (mean across participants)
    # and sem (from the variance  across participants) ======================

    m_times = np.mean(predictions, axis=2)
    m_epo = np.mean(m_times, axis=1)
    mean_plot = np.mean(m_epo, axis=0)
    sem_plot = np.std(m_epo, axis=0) / np.sqrt(n_subj)

    # ============== And now, let's plot ============================
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for k in range(n_cat):
        plt.plot(times, mean_plot[:, k], linewidth=linewidth)
        ax.set_ylim(ylim['distance'][n_cat])
        ax.fill_between(times, mean_plot[:, k] - sem_plot[:, k], mean_plot[:, k] + sem_plot[:, k], alpha=0.6)

    ax.axvline(0, 0, 200, color='k', linewidth=linewidth_zero)
    ax.set_xticks([np.round(0.433 * xx, 2) for xx in range(8)])
    for ti in [0.433 * xx for xx in range(8)]:
        ax.axvline(ti, 0, 200, color='k', linewidth=linewidth_other)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().tick_params(axis='both', which='major', labelsize=labelsize)
    plt.gca().set_xlabel('Testing Time (s)', fontsize=fontsize)
    plt.gca().set_ylabel('Probability ordinal position', fontsize=fontsize)

    return plt.gcf(), m_epo, times

# ----------------------------------------------------------------------------------------------------------------------
def load_predicted_ordinal_positions(analysis,control=False):
    """
    Function to load participants data (projections on the decision vector of the ordinal hyperplans) and to concatenate it.
    :param analysis: Analysis name
    :param prediction_type: 'distance'
    :param control: True if you run the control Fourier analyses
    :return: predictions, times
    """

    if control:
        data_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/' + '*_' + analysis + '_control.npy'
    else:
        data_path = config.result_path + '/decoding/decode_ordinal_position_fullblock/' + '*_' + analysis + '.npy'

    all_files = glob.glob(data_path)
    dists = []
    for file in all_files:
        res = np.load(file, allow_pickle=True).item()
        if "4seg" in file or "train4_test2" in file:
            print(" We are dealing with only 2 ordinal positions, so the output is 1D. We increase artificially the size of the distance score with 1-distance_ord_pos1")
            scores_2cat = np.asarray([np.asarray(res['distances']),1-np.asarray(res['distances'])])
            scores_2cat = np.moveaxis(scores_2cat,0,-1)
            dists.append(scores_2cat)
        else:
            dists.append(res['distances'])

    times = res['times']
    return np.asarray(dists), times

# ______________________________________________________________________________________________________________________
def compute_fft(signals,times):
    """
    Function to compute the Fast Fourier transform from the scipy fftpack
    """
    sig_fft = scipy.fftpack.fft(signals)
    sample_freq = scipy.fftpack.fftfreq(signals.shape[1], d=times[1]-times[0])
    return {'fft':sig_fft,'freqs':sample_freq}

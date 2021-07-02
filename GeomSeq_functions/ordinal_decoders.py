import glob
import mne
import numpy as np
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from jr.plot import pretty_gat, pretty_decod
import matplotlib.pyplot as plt

from GeomSeq_analyses import config
from GeomSeq_functions import utils, epoching_funcs
from GeomSeq_functions.primitive_decoding_funcs import gat_classifier_categories




# ------------- ------------- ------------- ------------- ------------- ------------- ------------- -------------  -----
# ------------- ------------- ------------- GAT FOR ORDINAL POSITION ---------- ------------- ------------- ------------
# ------------- ------------- ------------- ------------- ------------- ------------- ------------- -------------  -----

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

# ------------- ------------- ------------- ------------- ------------- ------------- ------------- -------------  -----
# ---- DECODING ORDINAL POSITION TRAINING THE ORDINAL DECODERS ON A WINDOW AND TESTING ON THE FULL 8 ITEMS SEQUENCE ----
# ------------- ------------- ------------- ------------- ------------- ------------- ------------- -------------  -----
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
def decode_ordinal_position_allBlocks(subject, baseline_or_not, PCA_or_not, sliding_window_or_not, micro_avg_or_not, control=False):
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



    baseline_or_not_seq = False

    epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not_seq, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',
        suffix='full_sequence')

    # _ _ _ _ _ concatenate back the sequence _ _ _ _ _
    epochs_for_decoding_full.crop(tmin=0,tmax=8*0.433)

    epo_full_4seg = epochs_for_decoding_full["sequence == '4segments'"]
    epo_full_4diag = epochs_for_decoding_full["sequence == '4diagonals'"]
    epo_full_2arcs = epochs_for_decoding_full["sequence == '2arcs'"]
    epo_full_2squ = epochs_for_decoding_full["sequence == '2squares'"]

    if control:
        epo_full_4seg._data = epochs_for_decoding_full["sequence == 'irregular'"]._data
        epo_full_4diag._data = epochs_for_decoding_full["sequence == 'repeat'"]._data
        epo_full_2arcs._data = epochs_for_decoding_full["sequence == 'repeat'"]._data
        epo_full_2squ._data = epochs_for_decoding_full["sequence == 'irregular'"]._data

    y_train4diag_test4seg = dec_4diag.predict_proba(back_to_miniblock(epo_full_4seg))
    y_train4seg_test4diag = dec_4seg.predict_proba(back_to_miniblock(epo_full_4diag))
    y_train2squares_test2arcs = dec_2squares.predict_proba(back_to_miniblock(epo_full_2arcs))
    y_train2arcs_test2squares = dec_2arcs.predict_proba(back_to_miniblock(epo_full_2squ))

    dist_train4diag_test4seg = dec_4diag.decision_function(back_to_miniblock(epo_full_4seg))
    dist_train4seg_test4diag = dec_4seg.decision_function(back_to_miniblock(epo_full_4diag))
    dist_train2squares_test2arcs = dec_2squares.decision_function(back_to_miniblock(epo_full_2arcs))
    dist_train2arcs_test2squares = dec_2arcs.decision_function(back_to_miniblock(epo_full_2squ))

    # ============== back to miniblock reorganizes the data into 4 (runs) X time series of the 12 repetitions ==================================

    times_1epo = epochs_for_decoding_full.times
    times = np.hstack([times_1epo+8*0.433*i for i in range(12)])

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

    suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not,
                           sliding_window_or_not=sliding_window_or_not)
    save_path = config.result_path + '/decoding/decode_ordinal_position_allBlocks/'
    utils.create_folder(save_path)
    np.save(save_path + subject + suffix + '__train4diag_test4seg'+suffix_control+'.npy', results_train4diag_test4seg)
    np.save(save_path + subject + suffix + '__train4seg_test4diag'+suffix_control+'.npy', results_train4seg_test4diag)
    np.save(save_path + subject + suffix + '__train2squares_test2arcs'+suffix_control+'.npy', results_train2squares_test2arcs)
    np.save(save_path + subject + suffix + '__train2arcs_test2squares'+suffix_control+'.npy', results_train2arcs_test2squares)






# ______________________________________________________________________________________________________________________
def decode_ordinal_position_allBlocks_train42_test24(subject, baseline_or_not, PCA_or_not, sliding_window_or_not, micro_avg_or_not, tmin=0.2, tmax=0.6,control=False):
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

    epochs_for_decoding, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq')


    # ==== select the epochs for the different training sequences ====

    epo_4SD = mne.concatenate_epochs([epochs_for_decoding["sequence == '4segments'"],epochs_for_decoding["sequence == '4diagonals'"]])
    epo_2AS = mne.concatenate_epochs([epochs_for_decoding["sequence == '2arcs'"],epochs_for_decoding["sequence == '2squares'"]])

    # ==== the training labels are the position in the the component ====
    label_4 = np.asarray([int(k) for k in epo_4SD.metadata["WithinComponentPosition"].values])
    label_2 = np.asarray([int(k) for k in epo_2AS.metadata["WithinComponentPosition"].values])
    # ====== training the decoders =======
    dec_4 = GeomSeq_funcs.decoding_funcs.decoding_funda.train_decoder_window(epo_4SD, label_4, tmin, tmax)
    dec_2 = GeomSeq_funcs.decoding_funcs.decoding_funda.train_decoder_window(epo_2AS, label_2, tmin, tmax)

    # ___________________________________________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    # ___________________________________________________________________________________________________________________________________

    baseline_or_not_seq = False

    epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not_seq, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',
        suffix='full_sequence')

    # _ _ _ _ _ concatenate back the sequence _ _ _ _ _
    epochs_for_decoding_full.crop(tmin=0,tmax=8*0.433)

    epo_4SD_full = mne.concatenate_epochs(
        [epochs_for_decoding_full["sequence == '4segments'"], epochs_for_decoding_full["sequence == '4diagonals'"]])
    epo_2AS_full = mne.concatenate_epochs(
        [epochs_for_decoding_full["sequence == '2arcs'"], epochs_for_decoding_full["sequence == '2squares'"]])


    y_train4_test2 = dec_4.predict_proba(back_to_miniblock(epo_2AS_full))
    y_train2_test4 = dec_2.predict_proba(back_to_miniblock(epo_4SD_full))

    dist_train4_test2 = dec_4.decision_function(back_to_miniblock(epo_2AS_full))
    dist_train2_test4 = dec_2.decision_function(back_to_miniblock(epo_4SD_full))

    # ============== back to miniblock reorganizes the data into 4 (runs) X time series of the 12 repetitions ==================================

    times_1epo = epochs_for_decoding_full.times
    times = np.hstack([times_1epo+8*0.433*i for i in range(12)])

    results_train4_test2= {'y_preds': np.asarray(y_train4_test2),
                                   'distances':dist_train4_test2,'times':times}
    results_train2_test4 = {'y_preds': np.asarray(y_train2_test4),
                                    'distances':dist_train2_test4,'times':times}


    suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not,
                           sliding_window_or_not=sliding_window_or_not)
    save_path = config.result_path + '/decoding/decode_ordinal_position_allBlocks/'
    utils.create_folder(save_path)
    np.save(save_path + subject + suffix + '__train4_test2.npy', results_train4_test2)
    np.save(save_path + subject + suffix + '__train2_test4.npy', results_train2_test4)


# ______________________________________________________________________________________________________________________
def decode_ordinal_position_allBlocks_repeat_irregular(subject, baseline_or_not, PCA_or_not, sliding_window_or_not, micro_avg_or_not, tmin=0.2, tmax=0.6):
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

    epochs_for_decoding, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq')

    epo_2arcs = epochs_for_decoding["sequence == '2arcs'"]

    epo_irregular = epochs_for_decoding["sequence == 'irregular'"]
    epo_repeat = epochs_for_decoding["sequence == 'repeat'"]

    # ==== the training labels are the position in the the component ====
    label_irregular = np.asarray([int(k) for k in epo_2arcs.metadata["WithinComponentPosition"].values])
    label_repeat = np.asarray([int(k) for k in epo_2arcs.metadata["WithinComponentPosition"].values])

    # ___________________________________________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    # ___________________________________________________________________________________________________________________________________

    baseline_or_not_seq = False
    epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not_seq, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',
        suffix='full_sequence')
    epochs_for_decoding_full.crop(tmin=0,tmax=8*0.433)

    y_pred_irregular = []
    y_pred_repeat = []
    distances_irregular = []
    distances_repeat = []

    for run in range(2,6):
        # ======== training on all the blocks and all the items appart from the ones belonging to run number "run"
        inds_train_irregular = np.where(epo_irregular.metadata["run_number"]!=run)[0]
        dec_irregular = GeomSeq_funcs.decoding_funcs.decoding_funda.train_decoder_window(epo_irregular[inds_train_irregular], label_irregular[inds_train_irregular], tmin, tmax)
        epo_irregular_full = back_to_miniblock(epochs_for_decoding_full["sequence == 'irregular' and run_number == %i"%run])
        distances_irregular.append(dec_irregular.decision_function(epo_irregular_full))
        y_pred_irregular.append(dec_irregular.predict_proba(epo_irregular_full))

        # _______________________________________________________________________________________
        inds_train_repeat = np.where(epo_repeat.metadata["run_number"]!=run)[0]
        dec_repeat = GeomSeq_funcs.decoding_funcs.decoding_funda.train_decoder_window(epo_repeat[inds_train_repeat], label_repeat[inds_train_repeat], tmin, tmax)
        epo_repeat_full = back_to_miniblock(epochs_for_decoding_full["sequence == 'repeat' and run_number == %i"%run])
        distances_repeat.append(dec_repeat.decision_function(epo_repeat_full))
        y_pred_repeat.append(dec_repeat.predict_proba(epo_repeat_full))

    # ============== back to miniblock reorganizes the data into 4 (runs) X time series of the 12 repetitions ==================================

    times_1epo = epochs_for_decoding_full.times
    times = np.hstack([times_1epo+8*0.433*i for i in range(12)])

    results_irregular = {'y_preds': np.asarray(y_pred_irregular),
                                   'distances':distances_irregular,'times':times}
    results_repeat = {'y_preds': np.asarray(y_pred_repeat),
                                    'distances':distances_repeat,'times':times}

    suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not,
                           sliding_window_or_not=sliding_window_or_not)
    save_path = config.result_path + '/decoding/decode_ordinal_position_allBlocks/'
    utils.create_folder(save_path)
    np.save(save_path + subject + suffix + '__irregular.npy', results_irregular)
    np.save(save_path + subject + suffix + '__repeat.npy', results_repeat)

# ___________________________ PLOTTING FUNCTION _____________________________________________________
def compute_decoding_results_ordinalposition(data_path,baseline_or_not,PCA_or_not,sliding_window_or_not,micro_avg_or_not,analysis,chance=1/11,tail=0, tmin = -0.2,tmax = 0.6,control=False,field_name=None):

    """
    Loads, computes the significance (over the full window) and plots the GAT performance
    :param data_path:
    :param baseline_or_not:
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param analysis:
    :param chance:
    :param tail:
    :param tmin:
    :param tmax:
    :param control:
    :param field_name:
    :return:
    """

    scores = load_scores_ordipos(data_path,baseline_or_not,PCA_or_not,sliding_window_or_not,micro_avg_or_not,analysis,control=control,field_name=field_name)

    train_times = np.linspace(start=tmin, stop=tmax, num=scores.shape[-1])
    print('n_time_points = %i'%scores.shape[-1])

    if chance ==0.25:
        clim = [0.23,0.27]
    elif chance == 0.5:
        clim = [0.48,0.52]

    scores_CBPT = np.asarray([scores[i] for i in range(len(scores))])
    print("==== the shape of scores_CBPT is =====")
    print(scores_CBPT.shape)
    print("==== Scores_CBPT is =====")
    print(scores_CBPT)

    sig = (GeomSeq_funcs.stat_funcs.stats(scores_CBPT-chance,tail=tail) < 0.05)

    mean_score = np.mean(scores, axis=0)
    pretty_gat(mean_score, times=train_times, chance=chance, sig=sig)
    fig_gat = plt.gcf()
    plt.close('all')
    pretty_decod(np.diagonal(scores,axis1 = 1,axis2=2), times=train_times, chance=chance)
    fig_diag = plt.gcf()
    plt.close('all')

    # ---- determine what is the peak value for the diagonal in the window 0 - 433 ms ----

    tmin_for_diag_max = 0.
    tmax_for_diag_max = 0.43
    filter_t_train = np.where(np.logical_and(train_times >= tmin, train_times <= tmax))[0]

    diag_values = np.mean(np.diagonal(scores,axis1 = 1,axis2=2),axis=0)
    diag_vals = diag_values[filter_t_train]
    times = train_times[filter_t_train]

    inds = np.argmax(diag_vals)
    max_val = diag_vals[inds]
    time_max = times[inds]
    print("The maximal value obtained in the window tmin = %0.02f and tmax = %0.02f for the analysis %s is %0.02f at time %0.03f"%(tmin_for_diag_max,tmax_for_diag_max,data_path,max_val,time_max))

    return mean_score, sig, fig_gat, fig_diag


# ------------- ------------- ------------- ------------- ------------- ------------- ------------- -------------  -----
# DECODING ORDINAL POSITION TRAINING THE ORDINAL DECODERS ON A WINDOW AND TESTING
#                                                       ON ALL THE BLOCK MADE OF THE 12 REPETITIONS OF THE SEQUENCE ----
# ------------- ------------- ------------- ------------- ------------- ------------- ------------- -------------  -----

def decode_ordinal_position_allBlocks_CV(subject, baseline_or_not, PCA_or_not, sliding_window_or_not,
                                         micro_avg_or_not, tmin=0.2, tmax=0.6,control=False):
    """
    This function trains the component number decoder on the data from one sequence type and tests it on the full mini-blocks
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

    epochs_for_decoding, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq')

    epo_4SD = mne.concatenate_epochs([epochs_for_decoding["sequence == '4segments'"],epochs_for_decoding["sequence == '4diagonals'"]])
    epo_2AS = mne.concatenate_epochs([epochs_for_decoding["sequence == '2arcs'"],epochs_for_decoding["sequence == '2squares'"]])

    if control:
        epo_for_data = mne.concatenate_epochs(
            [epochs_for_decoding["sequence == 'irregular'"], epochs_for_decoding["sequence == 'repeat'"]])
        epo_4SD._data = epo_for_data._data
        epo_2AS._data = epo_for_data._data

    # ==== the training labels are the position in the the component ====
    label_4SD = np.asarray([int(k) for k in epo_4SD.metadata["WithinComponentPosition"].values])
    label_2AS = np.asarray([int(k) for k in epo_2AS.metadata["WithinComponentPosition"].values])

    # ___________________________________________________________________________________________________________________________________
    # ==================== TEST ON THE FULL TIME WINDOW CORRESPONDING TO all the mini-blocks devoted to one sequence   ==================
    # ___________________________________________________________________________________________________________________________________

    baseline_or_not_seq = False
    epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
        subject, baseline_or_not_seq, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',
        suffix='full_sequence')
    epochs_for_decoding_full.crop(tmin=0,tmax=8*0.433)

    y_pred_4SD = []
    y_pred_2AS = []
    distances_4SD = []
    distances_2AS = []

    for run in range(2,6):
        # ======== training on all the blocks and all the items appart from the ones belonging to run number "run"
        inds_train_4SD = np.where(epo_4SD.metadata["run_number"]!=run)[0]
        dec_4SD = GeomSeq_funcs.decoding_funcs.decoding_funda.train_decoder_window(epo_4SD[inds_train_4SD], label_4SD[inds_train_4SD], tmin, tmax)

        # ======== re-concatenate the data corresponding to the mini-block 'run' ======
        epo_4seg_data = back_to_miniblock(epochs_for_decoding_full["sequence == '4segments' and run_number == %i"%run])
        epo_4diag_data = back_to_miniblock(epochs_for_decoding_full["sequence == '4diagonals' and run_number == %i"%run])

        if control:
            epo_4seg_data = back_to_miniblock(
                epochs_for_decoding_full["sequence == 'irregular' and run_number == %i" % run])
            epo_4diag_data = back_to_miniblock(
                epochs_for_decoding_full["sequence == 'repeat' and run_number == %i" % run])

        epo_full_4SD_data = np.vstack(np.asarray([epo_4seg_data,epo_4diag_data]))

        distances_4SD.append(dec_4SD.decision_function(epo_full_4SD_data))
        y_pred_4SD.append(dec_4SD.predict_proba(epo_full_4SD_data))

        # _______________________________________________________________________________________
        inds_train_2AS = np.where(epo_2AS.metadata["run_number"]!=run)[0]
        dec_2AS= GeomSeq_funcs.decoding_funcs.decoding_funda.train_decoder_window(epo_2AS[inds_train_2AS], label_2AS[inds_train_2AS], tmin, tmax)

        epo_2arc_data = back_to_miniblock(epochs_for_decoding_full["sequence == '2arcs' and run_number == %i"%run])
        epo_2squ_data = back_to_miniblock(epochs_for_decoding_full["sequence == '2squares' and run_number == %i"%run])

        if control:
            epo_2arc_data = back_to_miniblock(
                epochs_for_decoding_full["sequence == 'repeat' and run_number == %i" % run])
            epo_2squ_data = back_to_miniblock(
                epochs_for_decoding_full["sequence == 'irregular' and run_number == %i" % run])

        epo_full_2AS_data = np.vstack(np.asarray([epo_2arc_data,epo_2squ_data]))
        distances_2AS.append(dec_2AS.decision_function(epo_full_2AS_data))
        y_pred_2AS.append(dec_2AS.predict_proba(epo_full_2AS_data))
    # ============== back to miniblock reorganizes the data into 4 (runs) X time series of the 12 repetitions ==================================

    times_1epo = epochs_for_decoding_full.times
    times = np.hstack([times_1epo+8*0.433*i for i in range(12)])

    results_4SD = {'y_preds': np.asarray(y_pred_4SD),
                                   'distances':distances_4SD,'times':times}
    results_2AS = {'y_preds': np.asarray(y_pred_2AS),
                                    'distances':distances_2AS,'times':times}
    suffix_control = ''
    if control:
        suffix_control='_control'

    suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not,
                           sliding_window_or_not=sliding_window_or_not)
    save_path = config.result_path + '/decoding/decode_ordinal_position_allBlocks/'
    utils.create_folder(save_path)
    np.save(save_path + subject + suffix + '__4segments_4diagonals'+suffix_control+'.npy', results_4SD)
    np.save(save_path + subject + suffix + '__2arcs_2squares'+suffix_control+'.npy', results_2AS)

def back_to_miniblock(epochs_data):
    """
    Small function that concatenates back the epochs
    :param epochs_data:
    :return:
    """
    epochs_runs = []
    data = epochs_data.get_data()

    run_vals = np.unique(epochs_data.metadata['run_number'].values)

    for run_number in range(len(run_vals)):
        epochs_runs.append(np.hstack(data[run_number*12:(run_number+1)*12]))

    block_data = np.asarray(epochs_runs)

    return block_data

# ======================================================================================================================
# === DECODING THE COMPONENT POSITION IN SEQUENCE (AND NO LONGER THE ORDINAL POSITION WITHIN A COMPONENT) ==============
# ======================================================================================================================

def decode_component_position_4diagonals_4segments(subject):
    """ We decode the ordinal position of a subcomponent in the sequence. Here, subcomponents are made of 2 items.
    We train on 4segments and test on 4diagonals and vice-versa.
    Nothing special appart from this."""

    PCA_or_not = False
    sliding_window_or_not = True
    micro_avg_or_not = False

    for baseline_or_not in [True,False]:

        epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
            subject,baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',suffix = 'full_sequence')

        data_4segments, labels_4segments = get_component_position_in_sequence(epochs_for_decoding_full["sequence == '4segments'"])
        data_4diagonals, labels_4diagonals = get_component_position_in_sequence(epochs_for_decoding_full["sequence == '4diagonals'"])

        clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False))
        time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)
        time_gen.fit(data_4segments, labels_4segments)
        y_preds_train4segments = time_gen.predict(X=data_4diagonals)
        scores_train4segments = time_gen.score(data_4diagonals,labels_4diagonals)

        clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False))
        time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)
        time_gen.fit(data_4diagonals, labels_4diagonals)
        y_preds_train4diagonals = time_gen.predict(X=data_4segments)
        scores_train4diagonals = time_gen.score(data_4segments,labels_4segments)

        results = {'train_4segments_test_4diagonals':{},'train_4diagonals_test_4segments':{}}
        results['train_4segments_test_4diagonals'] = {'y_pred':y_preds_train4segments,'y_true':labels_4diagonals,'score':scores_train4segments}
        results['train_4diagonals_test_4segments'] = {'y_pred':y_preds_train4diagonals,'y_true':labels_4segments,'score':scores_train4diagonals}

        suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not=sliding_window_or_not)
        save_path = config.result_path+'/decoding/decode_component_position_in_sequence/'
        utils.create_folder(save_path)
        np.save(save_path+subject+suffix+'.npy',results)

# ======================================================================================================================
def decode_component_position_2arcs_2squares(subject):
    """
    We decode the ordinal position of a subcomponent in the sequence. Here, subcomponents are made of 4 items. We note
    that for 2arcs there is no clear beginning of the sequence. This is why we considered the trigonometrical and the
    order (i.e. from the presentation of the first sequence) cases.
    We train on 2arcs and test on 2squares and vice-versa.
    :param subject:
    :return:
    """

    PCA_or_not = False
    sliding_window_or_not = True
    micro_avg_or_not = False

    for baseline_or_not in [True,False]:

        epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
            subject,baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',suffix = 'full_sequence')

        data_2squares, labels_2squares = get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding_full["sequence == '2squares'"])
        data_2arcs_order, labels_2arcs_order = get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding_full["sequence == '2arcs'"])
        data_2arcs_trigo, labels_2arcs_trigo = get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding_full["sequence == '2arcs'"],segmentation='trigo')

        # ----- classifier from 2 squares tested on 2 arcs --------------------------------
        time_gen1 = GeneralizingEstimator(make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False)), scoring=None, n_jobs=8, verbose=True)
        time_gen1.fit(data_2squares, labels_2squares)
        y_preds_train2squares_test_2arcs_order = time_gen1.predict(X=data_2arcs_order)
        scores_train2squares_test_2arcs_order = time_gen1.score(data_2arcs_order,labels_2arcs_order)

        time_gen = GeneralizingEstimator(make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False)), scoring=None, n_jobs=8, verbose=True)
        time_gen.fit(data_2squares, labels_2squares)
        y_preds_train2squares_test_2arcs_trigo = time_gen.predict(X=data_2arcs_trigo)
        scores_train2squares_test_2arcs_trigo = time_gen.score(data_2arcs_trigo,labels_2arcs_trigo)

        # -----   classifier trained on the two types of 2 arcs and tested on 2squares  --------------------------------
        time_gen_2arcs_order = GeneralizingEstimator(make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False)), scoring=None, n_jobs=8, verbose=True)
        time_gen_2arcs_trigo = GeneralizingEstimator(make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False)), scoring=None, n_jobs=8, verbose=True)

        time_gen_2arcs_order.fit(data_2arcs_order,labels_2arcs_order)
        time_gen_2arcs_trigo.fit(data_2arcs_trigo, labels_2arcs_trigo)

        y_preds_train2arcs_order = time_gen_2arcs_order.predict(X=data_2squares)
        scores_train2arcs_order = time_gen_2arcs_order.score(data_2squares,labels_2squares)

        y_preds_train2arcs_trigo = time_gen_2arcs_trigo.predict(X=data_2squares)
        scores_train2arcs_trigo = time_gen_2arcs_trigo.score(data_2squares,labels_2squares)


        results = {'train_2squares_test2arcs_order':{},'train_2squares_test2arcs_trigo':{},
                   'train_2arcs_order_test2squares':{},'train_2arcs_trigo_test2squares':{}}

        results['train_2squares_test2arcs_order'] = {'y_pred':y_preds_train2squares_test_2arcs_order,'y_true':labels_2arcs_order,'score':scores_train2squares_test_2arcs_order}
        results['train_2squares_test2arcs_trigo'] = {'y_pred':y_preds_train2squares_test_2arcs_trigo,'y_true':labels_2arcs_trigo,'score':scores_train2squares_test_2arcs_trigo}
        results['train_2arcs_order_test2squares'] = {'y_pred':y_preds_train2arcs_order,'y_true':labels_2squares,'score':scores_train2arcs_order}
        results['train_2arcs_trigo_test2squares'] = {'y_pred':y_preds_train2arcs_trigo,'y_true':labels_2squares,'score':scores_train2arcs_trigo}

        suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not=sliding_window_or_not)
        save_path = config.result_path+'/decoding/decode_component_position_in_sequence/'
        utils.create_folder(save_path)
        np.save(save_path+subject+suffix+'_2arcs_2squares.npy',results)

# ======================================================================================================================
def decode_component_position_4segments_4diagonals_CV(subject):
    """We decode the ordinal position of a subcomponent in the sequence. Here, subcomponents are made of 2 items.
    We train on 4segments and 4diagonals data from all the runs appart one and test on the remaining one. We cross validate
    across the 4 runs."""
    PCA_or_not = False
    sliding_window_or_not = True
    micro_avg_or_not = False

    for baseline_or_not in [True,False]:

        epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
            subject,baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',suffix = 'full_sequence')

        print(" Now running for decode_component_position_4segments_4diagonals_CV the get_component_position_in_sequence that outputs also the runs")
        data_4segments, labels_4segments, runs_4segments = get_component_position_in_sequence(epochs_for_decoding_full["sequence == '4segments'"], return_runs=True)
        data_4diagonals, labels_4diagonals, runs_4diagonals = get_component_position_in_sequence(epochs_for_decoding_full["sequence == '4diagonals'"], return_runs=True)

        print(" Now running train_test_cv that outputs scores, y_preds, y_true")

        scores, y_preds, y_true = train_test_cv(data_4diagonals, data_4segments, labels_4diagonals, labels_4segments,
                                                runs_4diagonals, runs_4segments)

        print("Everything went fine, we save the results")
        results = {'train_test_4diagonals4segments':{}}
        results['train_test_4diagonals4segments'] = {'y_pred':np.asarray(y_preds),'y_true':np.asarray(y_true),'score':np.asarray(scores)}

        suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not=sliding_window_or_not)
        save_path = config.result_path+'/decoding/decode_component_position_in_sequence/'
        utils.create_folder(save_path)
        np.save(save_path+subject+suffix+'_train_test_4diagonals4segments.npy',results)

# ======================================================================================================================
def decode_component_position_2arcs_2squares_CV(subject):
    """
    We decode the ordinal position of a subcomponent in the sequence. Here, subcomponents are made of 4 items. We note
    that for 2arcs there is no clear beginning of the sequence. This is why we considered the trigonometrical and the
    order (i.e. from the presentation of the first sequence) cases.
    We train on 2arcs and 2squares data from all the runs appart one and test on the remaining one. We cross validate
    across the 4 runs. """
    PCA_or_not = False
    sliding_window_or_not = True
    micro_avg_or_not = False

    for baseline_or_not in [True,False]:

        epochs_for_decoding_full, results_decoding_path, micro_avg_or_not, sliding_window_or_not = load_epochs_and_apply_transformation(
            subject,baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not, filter='seq',suffix = 'full_sequence')

        data_2squares, labels_2squares, runs_2squares = get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding_full["sequence == '2squares'"], return_runs=True)
        data_2arcs_trigo, labels_2arcs_trigo, runs_2arcs_trigo = get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding_full["sequence == '2arcs'"],segmentation='trigo', return_runs=True)
        data_2arcs_order, labels_2arcs_order, runs_2arcs_order = get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding_full["sequence == '2arcs'"], return_runs=True)

        # ----- classifier from 2 squares tested on 2 arcs --------------------------------
        scores_2squares_2arcs_trigo, y_preds_2squares_2arcs_trigo, y_true_2squares_2arcs_trigo = train_test_cv(data_2squares, data_2arcs_trigo, labels_2squares, labels_2arcs_trigo, runs_2squares, runs_2arcs_trigo)
        scores_2squares_2arcs_order, y_preds_2squares_2arcs_order, y_true_2squares_2arcs_order = train_test_cv(data_2squares, data_2arcs_order, labels_2squares, labels_2arcs_order, runs_2squares, runs_2arcs_order)

        results = {'train_test_2squares2arcsorder':{},'train_test_2squares2arcstrigo':{}}

        results['train_test_2squares2arcstrigo'] = {'y_pred':y_preds_2squares_2arcs_trigo,'y_true':y_true_2squares_2arcs_trigo,'score':scores_2squares_2arcs_trigo}
        results['train_test_2squares2arcsorder'] = {'y_pred':y_preds_2squares_2arcs_order,'y_true':y_true_2squares_2arcs_order,'score':scores_2squares_2arcs_order}


        suffix = create_suffix(baseline_or_not, PCA_or_not, micro_avg_or_not, sliding_window_or_not=sliding_window_or_not)
        save_path = config.result_path+'/decoding/decode_component_position_in_sequence/'
        utils.create_folder(save_path)
        np.save(save_path+subject+suffix+'train_test_2arcs2squares.npy',results)

# ______________________________________________________________________________________________________________________
def train_test_cv(data_1, data_2, labels_1, labels_2, runs_1, runs_2):

    scores = []
    y_preds = []
    y_true = []
    for run in range(2, 5):
        idx_train_4segments = np.where(runs_2 == run)[0]
        idx_test_4segments = np.where(runs_2 != run)[0]
        idx_train_4diagonals = np.where(runs_1 == run)[0]
        idx_test_4diagonals = np.where(runs_1 != run)[0]

        data_train = np.vstack([data_2[idx_train_4segments], data_1[idx_train_4diagonals]])
        labels_train = np.hstack([labels_2[idx_train_4segments], labels_1[idx_train_4diagonals]])

        data_test = np.vstack([data_2[idx_test_4segments], data_1[idx_test_4diagonals]])
        labels_test = np.hstack([labels_2[idx_test_4segments], labels_1[idx_test_4diagonals]])

        clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=False))
        time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)
        time_gen.fit(data_train, labels_train)
        y_preds.append(time_gen.predict(X=data_test))
        y_true.append(labels_test)
        scores.append(time_gen.score(data_test, labels_test))

    return scores, y_preds, y_true

# ______________________________________________________________________________________________________________________
def get_component_position_in_sequence(epochs_for_decoding,return_runs = False):

    epochs_1 = epochs_for_decoding.copy().crop(tmin=0, tmax=2 * .433)
    epochs_2 = epochs_for_decoding.copy().crop(tmin=2 * .433, tmax=4 * .433)
    epochs_3 = epochs_for_decoding.copy().crop(tmin=4 * .433, tmax=6 * .433)
    epochs_4 = epochs_for_decoding.copy().crop(tmin=6 * .433, tmax=8 * .433)

    truncate = np.min([x.get_data().shape[2] for x in [epochs_1, epochs_2, epochs_3, epochs_4]])

    data = []
    labels = []
    runs = []
    for ii, epo in enumerate([epochs_1, epochs_2, epochs_3, epochs_4]):
        runs.append(epo.metadata['run_number'].values)
        data.append(epo._data[:, :, :truncate])
        labels.append([ii + 1] * len(epo))

    data = np.vstack(data)
    labels = np.hstack(labels)
    runs = np.hstack(runs)

    if return_runs:
        return data, labels, runs

    return data, labels

# ______________________________________________________________________________________________________________________
def get_component_position_in_sequence_2arcs_2squares(epochs_for_decoding,segmentation='onset', return_runs=False):

    """
    :param epochs_for_decoding: The epochs that will be reshaped in a way that labels the data in terms of the
    component position in a sequence.
    :param segmentation: 'onset' if you want to consider the order of the presentation or 'trigo' if you want to consider the
    direction of rotation.
    :return:
    """

    data = []
    labels = []
    runs = []

    if segmentation=='onset':
        epochs_1 = epochs_for_decoding.copy().crop(tmin=0, tmax=4 * .433)
        epochs_2 = epochs_for_decoding.copy().crop(tmin=4 * .433, tmax=8 * .433)
        truncate = np.min([x.get_data().shape[2] for x in [epochs_1, epochs_2]])
        for ii, epo in enumerate([epochs_1, epochs_2]):
            data.append(epo._data[:, :, :truncate])
            labels.append([ii + 1] * len(epo))
            runs.append(epo.metadata['run_number'].values)

    if segmentation=='trigo':
        for run in np.unique(epochs_for_decoding.metadata['run_number'].values[0]):
            epo = epochs_for_decoding["run_number == %i"%run]
            epochs_1 = epo.copy().crop(tmin=0, tmax=4 * .433)
            epochs_2 = epo.copy().crop(tmin=4 * .433, tmax=8 * .433)
            truncate = np.min([x.get_data().shape[2] for x in [epochs_1, epochs_2]])

            if 'V1' in epochs_1.metadata['sequence_subtype'].values[0]:
                for ii, epo in enumerate([epochs_1, epochs_2]):
                    data.append(epo._data[:, :, :truncate])
                    labels.append([ii + 1] * len(epo))
                    runs.append(epo.metadata['run_number'].values)
            elif 'V2' in epochs_1.metadata['sequence_subtype'].values[0]:
                for ii, epo in enumerate([epochs_1, epochs_2]):
                    data.append(epo._data[:, :, :truncate])
                    labels.append([-ii + 2] * len(epo))
                    runs.append(epo.metadata['run_number'].values)

            else:
                print("There is an error, neither V1 nor V2 are in the sequences subtypes")


    data = np.vstack(data)
    labels = np.hstack(labels)
    runs = np.hstack(runs)

    if return_runs:
        return data, labels, runs


    return data, labels


# ======================================================================================================================
# ================================== FOURIER TRANSFORM FUNCTIONS =======================================================
# ======================================================================================================================

# ______________________________________________________________________________________________________________________
def are_there_oscillations_in_ordinal_code(suffix='nobase_noPCA_SWoff_noma', prediction_type='distance', time_window=[300, 500],all_blocks=True,control=False):
    """
    This function computes the fourier transform of the ordinal code decoding results, averaging over the training time window
    :param suffix:
    :param prediction_type: 'distance' or 'ypred'. 'distance' is more sensitive.
    :param time_window: Training time window for which the decoder's output will be averaged.
    :param all_blocks: Set it to True if you want to compute the FFT on the predictions across the full 12 repetitions
    :param control:
    :return:
    """

    save_results_path = config.result_path + '/decoding/decode_ordinal_position_allBlocks/'
    save_fig_path = config.figure_path + '/decoding/decode_ordinal_position_allBlocks/'
    utils.create_folder(save_fig_path)

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


    # if all_blocks:
    #
    #
    # else:
    #     analyses = ['train4_test2', 'train2_test4']
        # analyses = ['train2squares_test2arcs', 'train2arcs_test2squares', 'train4seg_test4diag',
        #             'train4diag_test4seg','4segments_4diagonals','2arcs_2squares']
        # list_saving_names = analyses

    for ii, anal_list in enumerate(analyses):
        for anal in anal_list:
            print("====== running the analysis for %s =====" %anal)

            fig, signals, times = GeomSeq_funcs.decoding_funcs.decod_plot_funcs.average_and_plot_ord_pos(suffix=suffix, analysis_list=[anal],
                                           prediction_type=prediction_type, time_window=time_window,all_blocks=all_blocks,control=control)
            plt.close('all')
            n_subj, n_times, n_ord_positions = signals.shape
            print(" ---- there are %i subjects participating to this analysis "%n_subj)

            for ord_pos in range(n_ord_positions):
                fft_results = GeomSeq_funcs.fourier_funcs.compute_fft(signals[:,:,ord_pos],times)
                # ====================================================================================
                print("==== we are saving the fft results in here ======")
                if control:
                    save_path_cc = save_results_path+list_saving_names[ii]+suffix+'fft_results'+str(ord_pos+1)+'_control.npy'
                else:
                    save_path_cc = save_results_path+list_saving_names[ii]+suffix+'fft_results'+str(ord_pos+1)+'.npy'
                print(save_path_cc)
                # ====================================================================================
                np.save(save_path_cc,fft_results)

# ______________________________________________________________________________________________________________________
def load_scores_ordipos(data_path,baseline_or_not,PCA_or_not,sliding_window_or_not,micro_avg_or_not,analysis_list,field_name=None,control=False):
    """
    Loads the ordinal scores for the data in data_path and for the field_name in the results. Then it concatenates everything together.
    :param data_path:
    :param baseline_or_not:
    :param PCA_or_not:
    :param sliding_window_or_not:
    :param micro_avg_or_not:
    :param analysis_list:
    :param field_name:
    :param control:
    :return:
    """

    suff = GeomSeq_funcs.decoding_funcs.decod_utils.generate_file_suffix(data_path, baseline_or_not=baseline_or_not,
                                                                         PCA_or_not=PCA_or_not,
                                                                         micro_avg_or_not=micro_avg_or_not,
                                                                         sliding_window_or_not=sliding_window_or_not)

    print('The value of the control parameter is ')
    print(control)

    # ============= extract all the file names corresponding to the scores we want to analyze ==========================
    files = []
    for anal in analysis_list:
        if control:
            files.append(glob.glob(suff + '*' + anal + '_control.npy'))
        else:
            files.append(glob.glob(suff + '*' + anal + '.npy'))
    files = np.concatenate(files)
    print(files)
    print("======= THIS IS THE NUMBER OF FILES CONTRIBUTING TO THIS ANALYSIS =======\n")
    print(len(files))
    print("======= ======= ======= ======= ======= ======= ======= ========== =======\n")

    # ============ the following loop is made in order to be able to average the data coming from 2 different conditions
    # ============ for a given subject =================
    scores = []
    for subj in config.subjects_list:
        scores_subj = []
        for file in files:
            if subj in file:
                print("===== we are loading the data from %s "%file)
                data_subj = np.load(file, allow_pickle=True).item()
                if field_name is not None:
                    scores_subj.append(np.mean(np.asarray(data_subj[field_name]['scores']), axis=0))
                else:
                    scores_subj.append(np.mean(np.asarray(data_subj['scores']), axis=0))

        print("===== we are averaging the data for subject %s " % subj)
        scores_subj = np.asarray(scores_subj)
        scores_subj = np.mean(scores_subj,axis=0)
        scores.append(scores_subj)

    scores = np.asarray(scores)

    return scores



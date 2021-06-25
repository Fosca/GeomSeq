import pickle
from mne.decoding import GeneralizingEstimator
import numpy as np
import matplotlib.cm as cm
from sklearn.svm import SVC
import mne
from GeomSeq_functions import epoching_funcs
#MNE
#Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#JR tools
from jr.gat.scorers import  scorer_angle
from jr.gat.classifiers import AngularRegression
from sklearn.model_selection import KFold
from math import pi
from GeomSeq_analyses import config
import matplotlib.pyplot as plt
plt.register_cmap(name='viridis', cmap=cm.viridis)


def angular_decoder():
    """
    Builds an angular decoder
    """
    scaler = StandardScaler()
    model = AngularRegression()
    ang_decod = make_pipeline(scaler, model)
    return ang_decod

def decoder():
    """
    Builds a classifier decoder
    """
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    return clf


def load_localizer_decoder(subject,classifier=False, SW=None,step = None):
    """
    This function loads the localizer decoder that was computed.
    :param subject:
    :return:
    """
    if classifier:
        decoder_name = 'classifier_localizer.pkl'
    else:
        decoder_name = 'angular_localizer.pkl'

    save_folder = config.result_path+'/decoding/stimulus/'
    if SW is None:
        decoder_fullname = save_folder + subject + '/' + decoder_name
    else:
        if step is not None:
            decoder_fullname = save_folder + subject + '/SW_'+str(SW)+str(step) + decoder_name
        else:
            decoder_fullname = save_folder + subject + '/SW_'+str(SW) + decoder_name

    with open(decoder_fullname, 'rb') as fid:
        spatial_decoder = pickle.load(fid)

    return spatial_decoder

def build_localizer_decoder(subject,classifier=False,tmin = 0,tmax=0.5, SW=None,step = None,compute_cross_validation_score=False):
    """
    This function builds the localizer decoder from the data of the localizer part and the first items of the pairs in the primitive part.
    These are the two types of events for which the item's position on the screen cannot be anticipated.
    :param subject:
    :return:
    """
    if classifier:
        decoder_name = 'classifier_localizer.pkl'
    else:
        decoder_name = 'angular_localizer.pkl'

    save_folder = config.result_path+'/decoding/stimulus/'
    if SW is None:
        decoder_fullname = save_folder + subject + '/' + decoder_name
    else:
        if step is not None:
            decoder_fullname = save_folder + subject + '/SW_'+str(SW)+str(step) + decoder_name
        else:
            decoder_fullname = save_folder + subject + '/SW_'+str(SW) + decoder_name

    angle = np.arange(5*pi/8,5*pi/8-2*pi , -pi/4)

    if SW is not None:
        epochs_pairs = epoching_funcs.load_and_concatenate_epochs(subject,folder_suffix="anticipation", filter='pairs', no_rsa=True)
    else:
        epochs_pairs = epoching_funcs.load_and_concatenate_epochs(subject, filter='pairs', no_rsa=True)

    epochs_pairs_for_localizer = epochs_pairs["first_or_second == 1"]
    if 'ab' in subject:
        epochs_decoding_spatial_location = epochs_pairs_for_localizer
    else:
        if SW is not None:
            epochs_localizer = epoching_funcs.load_and_concatenate_epochs(subject, folder_suffix="anticipation",
                                                                      filter='localizer')
        else:
            epochs_localizer = epoching_funcs.load_and_concatenate_epochs(subject, filter='localizer')
        epochs_decoding_spatial_location = mne.concatenate_epochs([epochs_localizer, epochs_pairs_for_localizer])

    epochs_decoding_spatial_location = epochs_decoding_spatial_location["position_on_screen != 9"]

    if tmin is not None:
        epochs_decoding_spatial_location.crop(tmin=tmin,tmax=tmax)

    spatial_locations = epochs_decoding_spatial_location.metadata['position_on_screen'].values
    angular_spatial_locations = np.asarray([angle[ii - 1] for ii in spatial_locations])

    if SW is not None:
        if step is not None:
            epochs_decoding_spatial_location = epoching_funcs.sliding_window(epochs_decoding_spatial_location,
                                                                             sliding_window_size=SW,
                                                                             sliding_window_step=step)
        else:
            epochs_decoding_spatial_location = epoching_funcs.sliding_window(epochs_decoding_spatial_location,sliding_window_size=SW,sliding_window_step=1)

    X = epochs_decoding_spatial_location.get_data()
    if classifier:
        clf = decoder()
        spatial_decoder = GeneralizingEstimator(clf, n_jobs=8, scoring=None)
        y = spatial_locations
    else:
        clf = angular_decoder()
        spatial_decoder = GeneralizingEstimator(clf, n_jobs=8, scoring=scorer_angle)
        y = angular_spatial_locations

    if compute_cross_validation_score:
        scores = []
        X = epochs_decoding_spatial_location.get_data()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            spatial_decoder.fit(X_train, y_train)
            y_pred = spatial_decoder.predict(X_test)
            scores.append(scorer_angle(y_pred=y_pred, y_true=y_test))
        scores = np.asarray(scores)
        score = np.mean(scores, axis=0)
        np.save(decoder_fullname + '_score.npy',score)
    else:
        spatial_decoder.fit(X, y)
        with open(decoder_fullname,'wb') as fid:
            pickle.dump(spatial_decoder,fid)

    return True

def apply_localizer_to_sequences_8positions(subject):

    angles = np.asarray([5 * pi / 8, 3 * pi / 8, pi / 8, - pi / 8, -3 * pi / 8, -5 * pi / 8, -7 * pi / 8, 7 * pi / 8])
    scores = {'score_pos_%i':[] for i in range(1,9)}
    localizer = load_localizer_decoder(subject,classifier=False)
    epochs_sequences = epoching_funcs.load_and_concatenate_epochs(subject,filter='seq')
    epochs_sequences = epochs_sequences["sequence != 'memory1' or sequence != 'memory2'or sequence != 'memory4'"]
    for i in range(1,9):
        y_preds = localizer.predict(epochs_sequences["position_in_sequence ==%i"%i].get_data())
        y_true = [angles[pos] for pos in epochs_sequences.metadata['position_on_screen'].values]
        scores_pos = scorer_angle(y_true=y_true, y_pred=y_preds[:, :, :, 0])
        # we average across the different epochs (i.e. sequences, runs etc)
        # scores_pos = np.mean(scores_pos,axis=0)
        scores['score_pos_%i'%i] = scores_pos

    np.save(config.result_path+'/decoding/Figure3/PanelB/'+subject+'_scores.npy')


def apply_localizer_to_sequences(subject,classifier = True,tmin=-0.6,tmax=0.433, SW=None,step=None):
    """
    We apply the localizer decoder on the sequence events after the first sequence repetition and not on the violations
    :param subject:
    :return:
    """

    # ------ load the localizer ----

    if classifier:
        decoder_name = 'classifier_localizer.pkl'
    else:
        decoder_name = 'angular_localizer.pkl'

    save_folder = config.result_path+'/decoding/stimulus/'
    if SW is None:
        loca_fname = save_folder + subject + '/' + decoder_name
    else:
        if step is not None:
            loca_fname = save_folder + subject + '/SW_'+str(SW)+str(step) + decoder_name
        else:
            loca_fname = save_folder + subject + '/SW_'+str(SW) + decoder_name


    with open(loca_fname,'rb') as fid:
        localizer = pickle.load(fid)

    # ------ load the sequence epochs ----
    epochs_sequences , saving_suffix, micro_avg_or_not, sliding_window_or_not = epoching_funcs.load_epochs_and_apply_transformation(
        subject, filter='seq',
        baseline_or_not=False, PCA_or_not=False, micro_avg_or_not=False, sliding_window_or_not=False)

    epochs_sequences1 = epochs_sequences.copy().crop(tmin=tmin,tmax=tmax)
    epochs_sequences1 = epochs_sequences1["sequence != 'memory2' or sequence != 'memory4' or sequence != 'memory8' or sequence != '4diagonal' or sequence != '2crosses' and violation == 0"]

    suffix_SW = ''
    if SW is not None:
        epochs_sequences1 = epoching_funcs.sliding_window(epochs_sequences1,sliding_window_size=SW)
        suffix_SW = 'SW_' + str(SW)
        if step is not None:
            epochs_sequences1 = epoching_funcs.sliding_window(epochs_sequences1, sliding_window_size=SW,step=step)
            suffix_SW = 'SW_' + str(SW) + str(step)

    if classifier:
        y_preds1 = localizer.predict_proba(epochs_sequences1.get_data())
        dict_results1 = {'y_preds': y_preds1, 'metadata': epochs_sequences1.metadata,'times':epochs_sequences1.times}
        folder = config.result_path+'/decoding/stimulus/'+subject+'/'
        if SW is not None:
            folder = config.result_path+'/decoding/stimulus/'+subject+'/SW/'
            utils.create_folder(folder)
        np.save(folder+suffix_SW+'classifier_results_loca_on_seq'+str(round(tmin*1000))+'_'+str(round(tmax*1000))+'.npy',dict_results1)

    else:
        y_preds = localizer.predict(epochs_sequences1.get_data())
        dict_results = {'y_preds': y_preds, 'metadata': epochs_sequences.metadata,'times':epochs_sequences.times}
        np.save(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+'angular_results_loca_on_seq.npy',dict_results)



def decoding_successive_locations_in_sequence_Sebplot(subject_name, computer='CPU',compute_localizer=False):

    tmin = -0.2
    tmax = 1.
    decim = 4
    reject = None
    baseline = None
    predict_mode = 'mean-prediction'

    sequence_names , sequence_pos = foscafuncs.list_sequences_and_positions()
    sequence_names = sequence_names[:9]

    [data_path_sss, data_path_processed, data_path_behavioral_results, data_path_analysis_results, data_path_figures,
     data_path_stim_infos] = foscafuncs.define_paths(subject_name, computer)

    angles = np.asarray([5 * pi / 8, 3 * pi / 8, pi / 8, - pi / 8, -3 * pi / 8, -5 * pi / 8, -7 * pi / 8, 7 * pi / 8])


    saving_path_fig = data_path_figures + '/decoding/spatial_locations/sequences/regression/localizer_ang_90_190ms/'
    if computer == 'laptop':
        root = '/Volumes/NSdata/Geom_seq/'
    elif computer == 'CPU':
        root = '/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/'
    save_path = root + 'Article/results_for_figure/' + 'Figure3/' + 'spatial_decoding_as_function_of_position_Sebplot/'
    regression_fname = save_path + subject_name + '_localizer.pkl'

    if compute_localizer:
        foscafuncs.create_folder(saving_path_fig)

        X, y_cat, y_reg = epochselection.training_epochs(subject_name=subject_name, visualize_events=0, tmin=tmin,
                                                         tmax=tmax,
                                                         baseline=baseline, decim=decim, computer=computer)

        print('============ training localizer =============\n')

        localizer_angle = GATclassifiers.angular_decoder(predict_mode, X, y_reg)


        foscafuncs.create_folder(save_path)
        with open(regression_fname, 'wb') as fid:
            cPickle.dump(localizer_angle, fid)

        info_string = 'tmin = %.2f sec\n' % tmin + 'tmax = %.2f sec\n' % tmax + 'decim = %.2f \n' % decim + 'baseline = %s \n' % baseline
        info_funcs.save_info(save_path, 'info_localizer', info_string)

        print('============ localizer saved =============\n')

    else:
        with open(regression_fname, 'rb') as fid:
            localizer_angle = cPickle.load(fid)
        print('============ localizer loaded =============\n')
    # ============ apply localizer to successive positions in the sequence =============

    epochs_seq = epochselection.metadata_epochs_sequences(subject_name, tmin, tmax, reject=reject, baseline=baseline,
                                                          decim=decim, computer=computer)

    for i in range(9):
        seq_name = sequence_names[i]
        if i ==0:
            epo_seq = epochs_seq["sequence_ID == 'repeat+1' or sequence_ID == 'repeat-1' "]
        elif i<8:
            V1_seqname = seq_name + 'V1'
            V2_seqname = seq_name + 'V2'
            epo_seq = epochs_seq["sequence_ID == '%s' or sequence_ID == '%s' "%(V1_seqname,V2_seqname)]
        else:
            epo_seq = epochs_seq["sequence_ID == '%s'" % seq_name]

        save_path = root + 'Article/results_for_figure/' + 'Figure3/' + 'spatial_decoding_as_function_of_position_Sebplot/' +'/'+seq_name+'/'
        foscafuncs.create_folder(save_path)
        scores_dict = {'score_pos%i' % x: [] for x in range(1, 9)}

        print('============ predicting successive locations in sequence %s =============\n'%seq_name)

        for position in range(1, 9):
            print('============ localizer tested on position %i =============' % position)

            epochs_to_dec = epo_seq["position_in_sequence==%i" % (position)]
            y_pred = localizer_angle.predict(epochs_to_dec._data)
            y_true = [angles[pos] for pos in epochs_to_dec.metadata['position_on_screen'].values]
            score = scorer_angle(y_true=y_true, y_pred = y_pred[:,:,:, 0])
            scores_dict['score_pos%i' % position] = score

        np.save(save_path + subject_name + '_scores.npy', scores_dict)

    return True
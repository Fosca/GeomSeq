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
        decoder_name = 'classifier_localizer'
    else:
        decoder_name = 'angular_localizer'

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
        with open(decoder_fullname +'.pkl','wb') as fid:
            pickle.dump(spatial_decoder,fid)

    return True

def apply_localizer_to_sequences_8positions(subject):

    angles = np.asarray([5 * pi / 8, 3 * pi / 8, pi / 8, - pi / 8, -3 * pi / 8, -5 * pi / 8, -7 * pi / 8, 7 * pi / 8])
    scores = {'score_pos_%i'%i:[] for i in range(1,9)}
    localizer = load_localizer_decoder(subject,classifier=False)
    epochs_sequences = epoching_funcs.load_and_concatenate_epochs(subject,filter='seq')
    epochs_sequences = epochs_sequences["sequence != 'memory1' and sequence != 'memory2'and sequence != 'memory4'"]

    for i in range(1,9):
        y_preds = localizer.predict(epochs_sequences["position_in_sequence ==%i"%i].get_data())
        y_true = [angles[pos] for pos in epochs_sequences["position_in_sequence ==%i"%i].metadata['position_on_screen'].values]
        scores_pos = scorer_angle(y_true=y_true, y_pred=y_preds)
        scores['score_pos_%i'%i] = scores_pos

    np.save(config.result_path+'/decoding/stimulus/'+subject+'/scores_8positions.npy')


def apply_localizer_to_sequences(subject,classifier = True,tmin=-0.6,tmax=0.433, SW=None,step=None):
    """
    We apply the localizer decoder on the sequence events (except on the violations)
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

    epochs_sequences = epoching_funcs.load_and_concatenate_epochs(subject,filter = 'seq')
    # # ------ load the sequence epochs ----
    # epochs_sequences , saving_suffix, micro_avg_or_not, sliding_window_or_not = epoching_funcs.load_epochs_and_apply_transformation(
    #     subject, filter='seq',
    #     baseline_or_not=False, PCA_or_not=False, micro_avg_or_not=False, sliding_window_or_not=False)
    epochs_sequences1 = epochs_sequences.copy().crop(tmin=tmin,tmax=tmax)
    epochs_sequences1 = epochs_sequences1["sequence != 'memory1' or sequence != 'memory2' or sequence != 'memory4'"]
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
        np.save(folder+suffix_SW+'classifier_results_loca_on_seq'+str(round(tmin*1000))+'_'+str(round(tmax*1000))+'.npy',dict_results1)
    else:
        y_preds = localizer.predict(epochs_sequences1.get_data())
        dict_results = {'y_preds': y_preds, 'metadata': epochs_sequences.metadata,'times':epochs_sequences.times}
        np.save(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+'angular_results_loca_on_seq.npy',dict_results)

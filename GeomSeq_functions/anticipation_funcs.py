"""
Author: Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""


import numpy as np
from math import pi
from GeomSeq_analyses import config
from jr.plot import pretty_gat
import matplotlib.pyplot as plt
import pandas as pd

def angle_acc(y_pred, y_true):  # XXX note reversal of y_true & y_pred

    y_pred = np.asarray([y_pred[i] for i in range(len(y_pred))])
    y_true_exp = np.zeros((y_pred.shape))

    for k in range(y_pred.shape[1]):
        for l in range(y_pred.shape[2]):
            y_true_exp[:, k, l] = y_true
    angle_error = y_true_exp - y_pred
    scores = np.asarray(np.abs((angle_error + np.pi) % (2 * np.pi) - np.pi))

    return np.pi / 2 - scores

def binned(preds):
    """
    This function output
    :param preds: predicted angle from the output of the decoder
    :return:
    """
    preds = np.asarray([preds[i] for i in range(len(preds))])

    angle_bins = np.arange(5 * pi / 8+pi/8, 5 * pi / 8 - 2 * pi-pi/8, -pi / 4)

    binned_preds = np.empty(preds.shape)
    for i in range(len(preds)):
        preds_dig = preds[i,:,:]
        for k in range(preds_dig.shape[0]):
            preds_dig_k = preds_dig[k,:]
            binned_preds[i,k,:] = np.digitize(preds_dig_k, angle_bins)

    return binned_preds

def accuracy_score_per_epoch(ytrue,ypreds_epoch):
    """
    Returns a matrix of 0 and 1 of the same shape as ypreds, i.e. train X test times, where 0 is when ypred is different from y_true
    and 1 is when the two are equal
    :param ytrue:
    :param ypreds_epoch:
    :return:
    """
    return 1*(ypreds_epoch ==ytrue)


def from_prediction_loc_on_sequences_to_anticipation(subject,classifier=True,bin_results=False,SW=None):
    # TODO
    """
    Loads the predictions of the localizer applied to the sequences and computes the P2 and P2prime scores, and the anticipation.
    Saves all the results in a dataframe
    :param subject:
    :param classifier:
    :return:

    classifier = False
    bin_results = True
    """
    suffix_SW = ''

    if classifier:
        suffix = 'classifier_'
        list_suffix = ['-600_-500','-500_-400','-400_-300','-300_-200','-200_-100','-100_0','0_100','100_200','200_300','300_433']
        if SW is not None:
            list_suffix = ['-433_-300', '-300_-200', '-200_-100', '-100_0', '-200_-100']
            suffix_SW = 'SW/SW_'+str(SW)
        y_preds_all = []
        for suff in list_suffix:
            print(suff)
            dict_res = np.load(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+'classifier_results_loca_on_seq'+suff+'.npy',
                               allow_pickle=True).item()
            y_preds_all.append(dict_res['y_preds'])
        y_preds = np.concatenate(y_preds_all,axis=2)
    else:
        suffix = 'angular_'
        if bin_results:
            suffix = 'angular_binned_'

        dict_res = np.load(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+'angular_results_loca_on_seq.npy',
                               allow_pickle=True).item()
        y_preds = dict_res['y_preds']

    predictions_dataframe = dict_res['metadata']
    predictions_dataframe['y_preds'] = [y_preds[i] for i in range(y_preds.shape[0])]

    anticipation_dataframe = compute_scoresP2P2prime_and_anticipation(subject,predictions_dataframe,classifier,bin_results=bin_results)

    anticipation_dataframe = anticipation_dataframe.query("sequence != '2crosses' and sequence != '4diagonals' and sequence != 'memory1' and sequence != 'memory2' and sequence != 'memory4' and position_in_subrun > 8 ")
    filter_anticipation = []
    for k in range(len(anticipation_dataframe)):
        filter_anticipation.append(np.sum(anticipation_dataframe['anticipation'].values[k])!=0)
    anticipation_dataframe = anticipation_dataframe[filter_anticipation]

    anticipation_dataframe.to_pickle(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+suffix+'anticipation_dataframe_filter.pkl')

    return anticipation_dataframe

def load_and_suppress_8_first_items(subject,classifier=True,bin_results=False,SW=None):

    suffix_SW = ''
    if SW is not None:
        suffix_SW = 'SW_' + str(SW)
    if classifier:
        suffix = 'classifier_'
    else:
        suffix = 'angular_'
        if bin_results:
            suffix = 'angular_binned_'
    suffix = suffix_SW + suffix
    anticipation_dataframe = pd.read_pickle(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+suffix+'anticipation_dataframe_filter.pkl')
    anticipation_dataframe = anticipation_dataframe.query("position_in_subrun > 8")
    anticipation_dataframe.to_pickle(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix_SW+suffix+'anticipation_dataframe_without8first.pkl')




def compute_scoresP2P2prime_and_anticipation(subject,predictions_dataframe,classifier,bin_results=False):
    import numpy as np
    angle = np.arange(5 * pi / 8, 5 * pi / 8 - 2 * pi, -pi / 4)

    # ------ Define P2 and P2prime ---

    predictions_dataframe = predictions_dataframe.query("position_on_screen !=8")

    P2 = predictions_dataframe['position_on_screen'].values[1:]
    P1 = predictions_dataframe['position_on_screen'].values[:-1]
    diff_position = np.diff(predictions_dataframe['position_on_screen'].values)
    P2prime = P1 - diff_position
    P2prime = np.asarray([i % 8 for i in P2prime])

    #     # --- sanity checks that P2prime is well computed ---
    distances = np.asarray([0, 1, 2, 3, 4, 3, 2, 1])
    distanceP2P1 = distances[np.asarray([abs(y) for y in (P2 - P1)]) % 8]
    distanceP2primeP1 = distances[np.asarray([abs(z) for z in (P2prime - P1)]) % 8]
    if np.sum(distanceP2primeP1 != distanceP2P1)==0:
        print(" ==== EVERYTHING FINE IN THE COMPUTATION OF P2 AND P2PRIME =====")
    else:
        ValueError(" PROBLEM in the computation of the position of P2 and P2prime")
    if classifier:
        score_P2 = np.asarray([predictions_dataframe['y_preds'].values[i][:,:,P2[i-1]] for i in range(1,len(predictions_dataframe))])
        score_P2prime = np.asarray([predictions_dataframe['y_preds'].values[i][:,:,P2prime[i-1]] for i in range(1,len(predictions_dataframe))])
    else:
        if bin_results:
            ypreds = binned(predictions_dataframe['y_preds'].values)
            score_P2 = np.asarray([accuracy_score_per_epoch(P2[i-1],ypreds[i]) for i in range(1,len(predictions_dataframe))])
            score_P2prime = np.asarray([accuracy_score_per_epoch(P2prime[i-1],ypreds[i]) for i in range(1,len(predictions_dataframe))])
        else:
            P2_angle = np.asarray([angle[ii] for ii in P2])
            P2prime_angle = np.asarray([angle[ii] for ii in P2prime])
            score_P2 = np.asarray(angle_acc(predictions_dataframe['y_preds'].values[1:], P2_angle))
            score_P2prime = np.asarray(angle_acc(predictions_dataframe['y_preds'].values[1:], P2prime_angle))

    anticipation = score_P2 - score_P2prime

    anticipation_dataframe = predictions_dataframe[1:]
    anticipation_dataframe.drop(columns=['y_preds'])
    anticipation_dataframe['P2'] = [score_P2[i] for i in range(len(score_P2))]
    anticipation_dataframe['P2prime'] = [score_P2prime[i] for i in range(len(score_P2))]
    anticipation_dataframe['anticipation'] = [anticipation[i] for i in range(len(score_P2))]
    anticipation_dataframe['subjectID'] = [subject]*len(anticipation_dataframe)

    return anticipation_dataframe



def plot_anticipation_results(classifier=True,plot_results= True,bin_results=False,SW=None,subsample = None):

    suffix_SW = ''
    if SW is not None:
        suffix_SW = 'SW_' + str(SW)

    if classifier:
        suffix = 'classifier_'
        chance = 1 / 8
    else:
        suffix = 'angular_'
        chance = 0
        if bin_results:
            chance = 1 / 8
            suffix = 'angular_binned_'
    suffix = suffix_SW + suffix

    # load the data from all the participants
    scores_P2 = []
    scores_P2prime = []
    scores_anticipation = []
    for subject in config.subjects_list:
        print(subject)
        if SW is not None:
            print(config.result_path+'/decoding/stimulus/'+subject+'/SW/'+suffix+'anticipation_dataframe_filter.pkl')
            anticipation_df = pd.read_pickle(config.result_path+'/decoding/stimulus/'+subject+'/SW/'+suffix+'anticipation_dataframe_filter.pkl')
            if subsample is not None:
                anticipation_df = subsample(anticipation_df)
                anticipation_df.to_pickle(config.result_path+'/decoding/stimulus/'+subject+'/SW/'+suffix+'anticipation_dataframe_filter_sub.pkl')
        else:
            print(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix+'anticipation_dataframe_filter.pkl')
            anticipation_df = pd.read_pickle(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix+'anticipation_dataframe_filter.pkl')
        scores_P2.append(np.mean(anticipation_df['P2'],axis = 0))
        scores_P2prime.append(np.mean(anticipation_df['P2prime'],axis = 0))
        scores_anticipation.append(np.mean(anticipation_df['anticipation'],axis=0))
        del anticipation_df

    mean_scores_P2 = np.mean(scores_P2,axis= 0)
    mean_scores_P2prime = np.mean(scores_P2prime,axis= 0)
    mean_anticipation = np.mean(scores_anticipation,axis= 0)

    if classifier:
        test_times = [np.round(x,3) for x in np.linspace(-0.6,0.433,mean_scores_P2.shape[1])]
        train_times = np.linspace(0, 0.5, mean_scores_P2.shape[0])
        amp = 0.01
    else:
        test_times = [np.round(x,3) for x in np.linspace(-0.650,0.6,mean_scores_P2.shape[1])]
        train_times = np.linspace(0, 0.5, mean_scores_P2.shape[0])
        amp = 0.03
    if SW:
        test_times = [np.round(x,3) for x in np.linspace(-0.433,0.,mean_scores_P2.shape[1])]
        train_times = np.linspace(0.1, 0.2, mean_scores_P2.shape[0])

    print("The test times are:")
    print(test_times)

    if plot_results:
        # -------------------- now we plot ----------------
        pretty_gat(mean_scores_P2,times=train_times,test_times=test_times,chance=chance,clim=[chance-amp,chance+amp])
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+suffix+'score_P2_all.svg')
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+suffix+'score_P2_all.png')
        plt.close('all')
        pretty_gat(mean_scores_P2prime,times=train_times,test_times=test_times,chance=chance,clim=[chance-amp,chance+amp])
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+suffix+'score_P2prime_all.svg')
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+suffix+'score_P2prime_all.png')
        plt.close('all')
        pretty_gat(mean_anticipation,times=train_times,test_times=test_times,chance=0,clim=[-amp,+amp])
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+suffix+'anticipation.svg')
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+suffix+'anticipation.png')
        plt.close('all')

    results_average_epochs  = {'scores_P2':scores_P2,'scores_P2prime':scores_P2prime,'anticipation':scores_anticipation,'train_times':train_times,'test_times':test_times}
    np.save(config.result_path+'/decoding/stimulus/'+suffix+'results_scoresP2P2prime_average_epochs_all_subjects.npy',results_average_epochs)

    return results_average_epochs


def compute_significance_window_anticipation(classifier=True,tmin_avg=0.1,tmax_avg = 0.2, bin_results = False,SW=None):
    suffix_SW = ''
    if SW is not None:
        suffix_SW = 'SW_' + str(SW)

    if classifier:
        suffix = 'classifier_'
        chance = 1 / 8
    else:
        suffix = 'angular_'
        chance = 0
        if bin_results:
            chance = 1 / 8
            suffix = 'angular_binned_'
    suffix = suffix_SW + suffix

    results_average_epochs = np.load(config.result_path+'/decoding/stimulus/'+suffix+'results_scoresP2P2prime_average_epochs_all_subjects.npy',allow_pickle=True).item()
    train_times = results_average_epochs['train_times']
    test_times = results_average_epochs['test_times']
    scores_anticipation = np.asarray(results_average_epochs['anticipation'])

    inds_train = np.where(np.logical_and(train_times<=tmax_avg,train_times>=tmin_avg))[0]
    anticip = np.mean(scores_anticipation[:,inds_train,:],axis=1)

    times = np.asarray(test_times)
    for_permutation_Test = anticip[:,np.logical_and(times<0,times>-0.433)]
    significance_results = stat_funcs.stats(for_permutation_Test,tail=1)
    times_significance = False * np.ones([1, len(times)])
    times_significance[0][np.where(np.logical_and(times < 0, times > -0.433))[0]] = significance_results < 0.05

    # times_permutation = times[np.logical_and(times<0,times>-0.433)]
    print(" Anticipation is significant for time points : ")
    print(times[np.where(times_significance[0])[0]])
    # mean_anticip = np.mean(anticip[:,np.where(times_significance[0])[0]],axis=1)

    np.save(config.result_path +'/decoding/stimulus/' +suffix +'significant_timepoints.npy', times_significance[0])

    return times_significance[0]

def plot_average_P2_P2prime_anticipation(classifier=True,tmin_avg=0.1,tmax_avg = 0.2,bin_results=False,SW=None):

    suffix_SW = ''
    if SW is not None:
        suffix_SW = 'SW_'+str(SW)

    if classifier:
        suffix = 'classifier_'
        chance = 1/8
        vmin = 0.001
        vmax = 0.002
    else:
        suffix = 'angular_'
        chance = 0
        vmin = 0.01
        vmax = 0.042
        if bin_results:
            chance = 1/8
            suffix = 'angular_binned_'
    suffix = suffix_SW+suffix

    results_average_epochs = np.load(config.result_path+'/decoding/stimulus/'+suffix+'results_scoresP2P2prime_average_epochs_all_subjects.npy',allow_pickle=True).item()
    train_times = results_average_epochs['train_times']
    test_times = results_average_epochs['test_times']
    scores_anticipation = np.asarray(results_average_epochs['anticipation'])
    scores_P2prime = np.asarray(results_average_epochs['scores_P2prime'])
    scores_P2 = np.asarray(results_average_epochs['scores_P2'])

    inds_train = np.where(np.logical_and(train_times<=tmax_avg,train_times>=tmin_avg))[0]
    anticip_avg = np.mean(scores_anticipation[:,inds_train,:],axis=1)
    scoreP2prime_avg = np.mean(scores_P2prime[:,inds_train,:],axis=1)
    scoreP2_avg = np.mean(scores_P2[:,inds_train,:],axis=1)

    plot_nice_format(scoreP2_avg, test_times, chance, ymin=chance - vmin, ymax=chance + vmax,
                     save_name=suffix + 'P2_100200')
    plot_nice_format(scoreP2prime_avg, test_times, chance, ymin=chance - vmin, ymax=chance + vmax,
                     save_name=suffix + 'P2prime_100200')
    plot_nice_format(anticip_avg, test_times, chance=0, ymin=-vmin, ymax=vmax,
                     save_name=suffix + 'anticipation_mean100200')


def create_dataframe_avg_anticipation(subject,classifier=True,bin_results=False,tmin_avg = 0.1,tmax_avg=0.2,SW=None):
    """
    This function loads the results containing the scores from P2 and P2prime. It averages the anticipation over the significant time window
    and puts it in the shape of a dataframe which inherits the metadata of the epochs
    :param subject:
    :param classifier:
    :return:
    """

    suffix_SW = ''
    if SW is not None:
        suffix_SW = 'SW_'+str(SW)

    if classifier:
        suffix = 'classifier_'
        chance = 1/8
    else:
        suffix = 'angular_'
        chance = 0
        if bin_results:
            chance = 1/8
            suffix = 'angular_binned_'
    suffix = suffix_SW+suffix

    # ---- load the correct times of significance ------
    times_significance = np.load(config.result_path+'/decoding/stimulus/'+suffix+'significant_timepoints.npy')

    # ---- load the dataframe containing all the results ------
    print(subject)
    print(config.result_path + '/decoding/stimulus/' + subject + '/' + suffix + 'anticipation_dataframe_filter.pkl')
    if SW is not None:
        anticipation_dataframe = pd.read_pickle(
            config.result_path + '/decoding/stimulus/' + subject + '/SW/' + suffix + 'anticipation_dataframe_filter.pkl')
    else:
        anticipation_dataframe = pd.read_pickle(
            config.result_path + '/decoding/stimulus/' + subject + '/' + suffix + 'anticipation_dataframe_filter.pkl')
    # ---- compute from the score P2 and score P2prime the average anticipation over the training window tmin_avg, tmax_avg ------
    sP2 = np.asarray([anticipation_dataframe['P2'].values[i] for i in range(len(anticipation_dataframe))])
    sP2prime = np.asarray([anticipation_dataframe['P2prime'].values[i] for i in range(len(anticipation_dataframe))])
    santicipation = np.asarray([anticipation_dataframe['anticipation'].values[i] for i in range(len(anticipation_dataframe))])
    # ----- find the indices corresponding to the window for training times -----
    train_times = np.linspace(0, 0.5, sP2.shape[1])
    inds_train = np.where(np.logical_and(train_times <= tmax_avg, train_times >= tmin_avg))
    inds_train = inds_train[0]
    avg_P2 = np.mean(sP2[:, inds_train, :], axis=1)
    avg_P2prime = np.mean(sP2prime[:, inds_train, :], axis=1)
    avg_anticipation =  np.mean(santicipation[:, inds_train, :], axis=1)

    # ----- find the indices corresponding to the window for the significant testing times -----
    inds_test_significant = np.where(times_significance)[0]

    # ----- find the indices corresponding to the window for the significant testing times -----
    P2_wind = np.mean(avg_P2[:, inds_test_significant], axis=1)
    P2prime_wind = np.mean(avg_P2prime[:, inds_test_significant], axis=1)
    anticipation_wind =  np.mean(avg_anticipation[:, inds_test_significant], axis=1)

    anticipation_dataframe.drop(columns=['y_preds'])
    anticipation_dataframe['P2'] = P2_wind
    anticipation_dataframe['P2prime'] = P2prime_wind
    anticipation_dataframe['anticipation'] = anticipation_wind
    anticipation_dataframe['subjectID'] = [subject]*len(anticipation_dataframe)

    anticipation_dataframe.to_pickle(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix+'dataframe_average_anticipation.pkl')

    return anticipation_dataframe



def plot_nice_format(avg_P2,test_times,chance,ymin=None,ymax=None,save_name = 'classifier_P2_100200'):
    from jr.plot import pretty_gat, pretty_decod, pretty_slices

    if ymin is None:
        ymin = chance - 0.001
        ymax = chance + 0.004

    x_ticks = [np.round(i,1) for i in np.arange(round(np.min(test_times),1),round(np.max(test_times),1),0.1)]
    plt.close('all')
    y_ticks = [round(i,3) for i in np.linspace(ymin,ymax,5)]
    ylabels = [str(y_ticks[i]) for i in range(len(y_ticks))]
    xlabels = [int(x_ticks[i]*1000) for i in range(len(x_ticks))]
    pretty_decod(avg_P2,times = test_times,chance=chance,ax=plt.gca())
    plt.gca().set_ylim(ymin,ymax)
    plt.gca().set_yticks(y_ticks)
    plt.gca().set_xticks(x_ticks)
    plt.gca().set_yticklabels(ylabels,rotation='horizontal')
    plt.gca().set_xticklabels(xlabels,rotation='horizontal')
    plt.gca().axvline(-0.433, color='k', zorder=-3)
    plt.gca().axvline(0.433, color='k', zorder=-3)
    if save_name is not None:
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+save_name+'.svg')
        plt.gcf().savefig(config.figure_path + '/decoding/stimulus/'+save_name+'.png')

    return plt.gcf()


def concatenate_dataframes_and_save_csv(classifier,bin_results, SW=None):
    suffix_SW = ''
    if SW is not None:
        suffix_SW = 'SW_' + str(SW)
    if classifier:
        suffix = 'classifier_'
    else:
        suffix = 'angular_'
        if bin_results:
            suffix = 'angular_binned_'
    suffix = suffix_SW + suffix

    dataframe = []
    for subject in config.subjects_list:
        print(subject)
        dataf = pd.read_pickle(
            config.result_path + '/decoding/stimulus/' + subject + '/' + suffix + 'dataframe_average_anticipation.pkl')
        distances = np.hstack([np.nan, np.diff(dataf['position_on_screen'].values)])
        df = dataf[['anticipation', 'sequence', 'position_in_sequence', 'subjectID', 'position_in_subrun']]
        df['distance_to_previous'] = distances
        dataframe.append(df)
        del dataf
    dataframes = pd.concat(dataframe)

    compl = []
    emp_comp = []
    subject_number = []

    complexity = {"repeat": 5, "alternate": 7, "4segments": 7, "2arcs": 8, "2squares": 8, "2rectangles": 10,
                  "irregular": 16}
    empirical_complexity = {"repeat": -1.4108, "alternate": 0.6311, "4segments": -1.1167, "2arcs": -0.854,
                            "2squares": -0.2317, "2rectangles": 0.5455, "irregular": 1.7301}
    all_subj = config.subjects_list

    for k in range(len(dataframes)):
        table_line = dataframes.iloc[k]
        compl.append(complexity[table_line['sequence']])
        emp_comp.append(empirical_complexity[table_line['sequence']])
        subject_number.append(all_subj.index(table_line['subjectID']) + 1)

    dataframes['Complexity'] = compl
    dataframes['Empirical_complexity'] = emp_comp
    dataframes['Subject_number'] = subject_number

    dataframes.to_csv(config.result_path + "decoding/stimulus/" + suffix + "anticipation_table.csv")


def extract_anticipation_per_sequence(classifier=True,bin_results=False,SW=None):
    suffix = ''
    if SW:
        suffix += 'SW_'+str(SW)
    if classifier:
        suffix += 'classifier_'
    else:
        suffix += 'angular_'
        if bin_results:
            suffix += 'binned_'

    scores_anticipation = {'repeat':[],'alternate':[],'4segments':[],'2arcs':[],'2squares':[],'2rectangles':[],'irregular':[]}
    for subject in config.subjects_list:
        print(subject)
        if SW is not None:
            anticipation_df = pd.read_pickle(config.result_path+'/decoding/stimulus/'+subject+'/SW/'+suffix+'anticipation_dataframe_filter.pkl')
        else:
            anticipation_df = pd.read_pickle(config.result_path+'/decoding/stimulus/'+subject+'/'+suffix+'anticipation_dataframe_filter.pkl')
        for seqID in scores_anticipation.keys():
            anticipation_df_seq = anticipation_df.query("sequence == '%s'"%seqID)
            scores_anticipation[seqID].append(np.mean(anticipation_df_seq['anticipation'],axis=0))
        del anticipation_df

    np.save(config.result_path+'/decoding/stimulus/'+suffix+'_avg_sequenceID.npy',scores_anticipation)

    return True

def plot_anticipation_per_seqID(classifier=True,bin_results=False,SW=None,tmin_avg = 0.1,tmax_avg=0.2, ymin=None, ymax=None):

    suffix = ''
    if SW:
        suffix += 'SW_'+str(SW)
    if classifier:
        suffix += 'classifier_'
    else:
        suffix += 'angular_'
        if bin_results:
            suffix += 'binned_'


    scores_anticipation = np.load(config.result_path+'/decoding/stimulus/'+suffix+'_avg_sequenceID.npy',allow_pickle=True).item()

    for seqID in scores_anticipation.keys():
        plt.close('all')
        print("---- running for seqID = %s"%seqID)

        scores_seq = np.asarray(scores_anticipation[seqID])

        if classifier:
            test_times = [np.round(x, 3) for x in np.linspace(-0.6, 0.433, scores_seq.shape[2])]
            train_times = np.linspace(0, 0.5, scores_seq.shape[1])
        else:
            test_times = [np.round(x, 3) for x in np.linspace(-0.650, 0.6, scores_seq.shape[2])]
            train_times = np.linspace(0, 0.5, scores_seq.shape[1])
        if SW:
            test_times = [np.round(x, 3) for x in np.linspace(-0.433, 0., scores_seq.shape[2])]
            train_times = np.linspace(0.1, 0.2, scores_seq.shape[1])

        inds_train = np.where(np.logical_and(train_times <= tmax_avg, train_times >= tmin_avg))[0]
        scores_seq_avg = np.mean(np.asarray(scores_seq)[:, inds_train, :], axis=1)
        plot_nice_format(scores_seq_avg, test_times, chance = 0, ymin=ymin, ymax=ymax, save_name='/sequences/'+suffix+seqID)









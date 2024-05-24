# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:15:27 2022
This module regroups all functions for Hexoskin project.
The detection of epileptic seizures using a connected shirt
https://doi.org/10.1111/epi.18021

@author: Oumayma Gharbi
"""

# import utils
from utils import *
from .ACM_features import *
from .ECG_features import *
from .config import home


# pylint: disable = invalid-name
# pylint: disable = unnecessary-semicolon

folder = home + 'Oumayma/Hexoskin/'
records = pd.read_csv(folder + 'updateJULY2023_records_details_with_seizures.csv')  # 29 juill 2023

out_path = f'Oumayma/Hexoskin/NestedCV/FBTCS_patients/'
t1 = datetime.datetime.now()
tmp = datetime.datetime.strftime(t1, '%Y%m%d_%H%M%S')
log_file = home + out_path + f'/draft_log_files/test_logfile_nestedCV_log_{tmp}.log'

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def plot_ECG_ACM_seizure(df, edf_info, ecg, acc_x, acc_y, acc_z, seizure_row,
                         margin=30, save=False):
    """
    This function plots ECG and ACM from hexoskin record during a specified annotated seizure

        df : Dataframe containing mapping between seizures details and corresponding EDF record
        seizure_row : selected seizure row from df dataframe with respect to patient_ID
        edf_info, ecg, acc_x, acc_y, acc_z are extracted from EDF file
        margin : display margin in seconds
    """
    fs = int(edf_info['n_samps'][0])

    s_start = df.Seizure_Start_UTC[seizure_row]
    s_end = df.Seizure_End_UTC[seizure_row]

    diff_start = datetime.datetime.fromisoformat(s_start) - edf_info['meas_date']
    diff_end = datetime.datetime.fromisoformat(s_end) - edf_info['meas_date']

    hh_0, mm_0, ss_0 = 0, 0, int(diff_start.total_seconds())-margin
    hh_1, mm_1, ss_1 = 0, 0, int(diff_end.total_seconds())+margin

    start = (hh_0*3600 + mm_0*60 + ss_0)*fs
    end = (hh_1*3600 + mm_1*60 + ss_1)*fs
    duration = (end-start)/fs

    x_axis = np.linspace(0, duration, len(ecg[start:end]))
    print('seizure duration is ', duration-2*margin)   # removing margin for display

    Patient_ID = df.Patient_ID[seizure_row]
    Record_name = df.Record_name[seizure_row]
    S_Class = df.Seizure_Classification[seizure_row]

    # pylint: disable = unused-variable
    fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(20, 9))
    txt = 'ECG and ACM from a patient before and during an epileptic seizure, p'
    ax1.set_title(txt+str(Patient_ID)+', '+Record_name+', '+S_Class+' seizure '+str(s_start))
    ax1.set(xlim=[0, duration])
    ax2.set(xlim=[0, duration])
    ax1.yaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_xlabel('time (s)')
    ax1.plot(x_axis, ecg[start:end], label='ECG')
    ax1.axvline(margin, label='seizure onset', c='r')
    ax1.axvline(duration-margin, label='seizure offset', c='r')
    ax1.legend(bbox_to_anchor=(1.01, 0.98), loc='upper left', borderaxespad=0.)

    ax2.plot(x_axis, acc_x[start:end], label='ACM - x', c='orange')
    ax2.plot(x_axis, acc_y[start:end], label='ACM - y', c='g')
    ax2.plot(x_axis, acc_z[start:end], label='ACM - z', c='darkred')
    ax2.axvline(margin, label='seizure onset', c='r')
    ax2.axvline(duration-margin, label='seizure offset', c='r')
    plt.legend(bbox_to_anchor=(1.01, 0.98), loc='upper left', borderaxespad=0.)
    if save:
        s = s_start.replace(':', '').replace(' ', '_')
        path = home + '/Oumayma/Figures/seizure_p'
        plt.savefig(path+str(Patient_ID)+'_'+s+'_'+S_Class+'_ECG-triACM26.png',
                    dpi=400, bbox_inches='tight')


def seizure_onset_offset_duration_in_seconds(df, seizure_row, record_edf_info):
    """
    Given the time in ISO format HH:MM:SS, return time in seconds
    (related to beginning of recording
    """
    s_start = df.Seizure_Start_UTC[seizure_row]
    s_end = df.Seizure_End_UTC[seizure_row]
    diff_start = datetime.datetime.fromisoformat(s_start) - record_edf_info['meas_date']
    diff_end = datetime.datetime.fromisoformat(s_end) - record_edf_info['meas_date']
    hh_0, mm_0, ss_0 = 0, 0, int(diff_start.total_seconds())
    hh_1, mm_1, ss_1 = 0, 0, int(diff_end.total_seconds())
    start = (hh_0*3600 + mm_0*60 + ss_0)
    end = (hh_1*3600 + mm_1*60 + ss_1)
    duration = (end-start)
    return [start, end, duration]


def annot_index(seizures_df, seizure_row, record_df, edf_info):
    """
    Given the seizure row from the seizures dataframe,
    return the corresponding row in records dataframe
    """
    r = seizures_df.Record_name[seizure_row]
    temp_df = record_df[record_df.Record_name == r]
    record_Onsets = temp_df.Onsets.values[0][1:-1]
    record_Onsets = np.fromstring(record_Onsets, dtype=int, sep=',')

    seizure_onset = seizure_onset_offset_duration_in_seconds(seizures_df, seizure_row, edf_info)[0]

    return np.where(record_Onsets == seizure_onset)[0][0]


def detect_Rpeaks(raw_py, start, end, step=10):
    """
    Detect R peaks from QRS complex in the ECG signal of an EDF rawpy object.
    raw_py : raw edf data
    step = 10 sec default (window size for finding ecg events)
    """

    R_peaks = np.array([], dtype=int)
    R_peaks_val = np.array([], dtype=int)

    ecg = raw_py['4113:ECG_I'][0][0]
    event_id = 999

    for i in tqdm(range(start, end, step)):

        raw_crop = raw_py.copy().crop(tmin=i, tmax=i+step)
        ecg_events_crop, _, _ = mne.preprocessing.find_ecg_events(raw_crop, event_id,
                                                                  ch_name='4113:ECG_I',
                                                                  filter_length='5s')  # ;

        R_peaks_crop = ecg_events_crop[:, 0]
        R_peaks_val_crop = [ecg[r] for r in R_peaks_crop]

        R_peaks = np.concatenate((R_peaks, R_peaks_crop))
        R_peaks_val = np.concatenate((R_peaks_val, R_peaks_val_crop))
    return R_peaks, R_peaks_val


def recording_before_onset(df_seizures, df_seizures_row, df_records, N=1):
    """
    Verify availability of N hours prior to seizure onset on the record.
    Args:
        df_seizures : seizures dataframe
        df_seizures_row : row in seizures dataframe
        df_records : Record details dataframe
        N: recording time desired before seizure onset, in hours
    Returns:
        bool
    """
    r = df_seizures.iloc[df_seizures_row]['Record_name']
    s_start = df_seizures.iloc[df_seizures_row]['Seizure_Start_UTC']
    r_start = df_records[df_records['Record_name'] == r].Record_start_date.values[0]

    s_start = datetime.datetime.strptime(s_start, '%Y-%m-%d %H:%M:%S%z')
    r_start = datetime.datetime.strptime(r_start, '%Y-%m-%d %H:%M:%S%z')

    return (s_start - r_start).total_seconds() > N*3600


def interictal_before_onset(temp_annot, seizure_onset, interictal_period=1):
    """
    Verify availability of N hours prior to seizure onset on the record.
    Args:
        temp_annot: annotation of the seizure/interictal segments
        seizure_onset
        interictal_period: interictal time desired before seizure onset, in hours
    Returns:
        bool
    """
    it_period = int(interictal_period*3600)
    if seizure_onset-it_period > 0:
        cumul = np.sum(temp_annot[seizure_onset-it_period:seizure_onset-1])
        return cumul < 5
    else:
        return False


def generate_temp_record_annot(df_records, record_name):
    temp_df = df_records[df_records.Record_name == record_name]

    Onsets = temp_df.Onsets.values[0][1:-1]
    Onsets = np.fromstring(Onsets, dtype=int, sep=',')

    Durations = temp_df.Durations.values[0][1:-1]
    Durations = np.fromstring(Durations, dtype=int, sep=',')

    Descriptions = temp_df.Descriptions.values[0][1:-1]
    Descriptions = Descriptions.replace("'", "").split(", ") if not (pd.isna(Descriptions)) else []

    hh, mm, ss = np.fromstring(temp_df.Record_duration.values[0], dtype=int, sep=':')
    record_duration = hh*3600 + mm*60 + ss

    temp_annot = np.zeros(record_duration)
    for i in range(len(Onsets)):
        if Descriptions[i] == 'FBTCS':
            on = Onsets[i]
            d = Durations[i]
            temp_annot[on:on+d] = 1
    return temp_annot


def firing_power(pre_input, tau=6, threshold=0.85):
    """
    Implementation of the firing power output regularization method
    described in Teixeira et al. (2012), which minimizes the number
    of false alarms raised due to noise.

    Teixeira C, Direito B, Bandarabadi M, Dourado A. Output regularization
    of SVM seizure predictors: Kalman Filter versus the "Firing Power" method.
    Annu Int Conf IEEE Eng Med Biol Soc. 2012;2012:6530-3.
    doi: 10.1109/EMBC.2012.6347490. PMID: 23367425.

    Args:
        pre_input: Binary prediction output of the classifier to be post-processed.
        tau: time window τ indication number of predictions to be averaged
        (6 time windows for an extraction window of 15s and an overlap of 10s includes 40s)
        threshold: threshold value of minimal fraction of full firing power
        (if at least 85% of the predictions in the sliding window predict an event,
        then output is set to 1)

    Returns:
        FPow : array of regularized predictions representing high-risk alarms
    """

    FPow = []
    try:
        assert tau > 1
        if len(pre_input) > tau:
            # in order to have FP-output array with same shape as input predictions
            FPow = list(np.zeros(tau-1))
            for i in range(tau, len(pre_input)+1, 1):
                FP_n = np.sum(pre_input[i - tau:i]) / tau
                # print(i, i+tau, pre_input[i:i+tau], FP_n)
                FPow.append(FP_n)
        else:
            FPow = pre_input
    except AssertionError:
        logger.error("Provide τ value > 1 to apply firing power method.", exc_info=True)
        if tau == 1:
            FPow = pre_input
            logger.info("For τ=1, returned same input array.")
    return FPow


def legend_without_duplicate_labels(figure):
    """ Remove duplicate labels from a legend """
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(),
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)


def predictions_per_seizure(test_dataset, seizure_row, model, scaled_X_test,
                            tau=6, threshold=0.85,
                            plot_alarms=True, show_plots=True, save_fig=None,
                            home_path=home):
    """
    Generate predictions on the specified seizure given a trained model,
    and calculate performance metrics.
    Specified seizure implies one preictal hour and the specified seizure.
    Args:
        test_dataset : data with test' seizures
        seizure_row  : seizure index in data
        model        : trained model to evaluate
        scaled_X_test: test features scaled using mean of train data
        tau          : window size for firing power regularization
        threshold    : threshold for firing power regularization
        plot_alarms  : bool, plot vertical lines when firing power average crosses the threshold
        show_plots   : bool, plot features and predictions for 1h preictal of the selected seizure
        save_fig     : str, default=None, is path provided, save the plots as png.
        home_path    : str, home directory for the EDF and features

    Returns:
        dictionary with performances measures of the selected seizure.
    {predictions, regularized_predictions, alarms_idx,
    detection_latency, nb_false_alarms, nb_missed_alarms}
    """

    data_path = home_path + 'Detection Multimodale Non-invasive/Donnees patients/Hexoskin/New_Data_EDF/'
    seizures = pd.read_csv(folder + 'seizures-details_mapping_interictal_availability_update.csv')

    # X_test = test_dataset.iloc[:, 4:-1]
    # y_test = test_dataset.iloc[:, -1]
    predictions = model.predict(scaled_X_test)

    fs = 256
    test_sz = test_dataset[test_dataset.Seizure_index == seizure_row]
    record = test_dataset[test_dataset.Seizure_index == seizure_row]['Record_name'].unique()[0]
    p = test_dataset[test_dataset.Seizure_index == seizure_row]['Patient_ID'].unique()[0]

    print('\npatient %a, seizure %s' % (p, seizure_row))
    print()

    seizure_onset, seizure_duration, diff = 0, 0, 0
    sz_0, sz_1 = 0, 0
    tm = 0
    df_acm, df_hrv = pd.DataFrame(), pd.DataFrame()

    for directory in os.listdir(data_path):
        if 'p' + str(p) in directory:
            edf_path = data_path + directory + '/' + record
            print(edf_path)

            info, edf_info, orig_units = mne.io.edf.edf._get_info(edf_path, stim_channel='auto',
                                                                  eog=None, misc=None,
                                                                  exclude=(), infer_types=True,
                                                                  preload=False)
            annotations = mne.read_annotations(edf_path[:-4] + '_annotations_FBTCS_FAS_FIAS-annot.txt')
            s_index = np.where(annotations.onset == seizure_onset_offset_duration_in_seconds(seizures,
                                                                                             seizure_row,
                                                                                             edf_info)[0])[0][0]
            seizure_onset = annotations[s_index]['onset']
            seizure_duration = annotations[s_index]['duration']

            sz_dt = datetime.datetime.combine(edf_info['meas_date'].date(), datetime.datetime.min.time())
            sz_dt = sz_dt.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
            diff = edf_info['meas_date'] - sz_dt
            tm = test_sz['timestamp']
            tm = np.array(tm) / fs + int(diff.total_seconds())
            hrv_col = [x for x in test_sz.columns if 'HRV' in x]
            acm_col = [x for x in test_sz.columns if 'Gait' in x]
            df_acm = test_sz[acm_col]
            df_hrv = test_sz[hrv_col]

            sz_idx = test_sz.index.tolist()
            test_lst = test_dataset.index.tolist()
            sz_0 = test_lst.index(sz_idx[0])
            sz_1 = test_lst.index(sz_idx[-1])

    # regularize prediction output with firing power method
    reg_pred = firing_power(predictions[sz_0:sz_1+1], tau=tau)  # , threshold=threshold)

    if save_fig:
        save_fig = save_fig + '_p'+str(p) + '_' + str(seizure_row)
        plot_ACM_features(tm, df_acm, seizure_onset, seizure_onset + seizure_duration, diff,
                          show=show_plots,
                          save_fig=save_fig + '_acm.png')
        plot_tHRV_with_timestamp(tm, df_hrv, seizure_onset, seizure_onset + seizure_duration, diff,
                                 show=show_plots,
                                 save_fig=save_fig + 'HRVt.png')
        plot_nlHRV_with_timestamp(tm, df_hrv, seizure_onset, seizure_onset + seizure_duration, diff,
                                  show=show_plots,
                                  save_fig=save_fig + '_HRVnl.png')
    elif show_plots:
        plot_ACM_features(tm, df_acm, seizure_onset, seizure_onset + seizure_duration, diff)
        plot_tHRV_with_timestamp(tm, df_hrv, seizure_onset, seizure_onset + seizure_duration, diff)
        plot_nlHRV_with_timestamp(tm, df_hrv, seizure_onset, seizure_onset + seizure_duration, diff)
    else:
        pass

    a = np.array(reg_pred) - threshold
    asc_zc = np.where(np.diff(np.sign(a)) > 0)[0]  # crossing threshold in an ascending way

    if show_plots:
        formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.set(xlabel="Time", title="True vs predicted labels for patient %a, seizure %s" % (p, seizure_row))
        ax.plot(tm, test_sz['annotation'], c='g', label='y_true')
        ax.scatter(tm, predictions[sz_0:sz_1+1], c='magenta', label='y_pred')
        ax.plot(tm, reg_pred, marker='.', label='y_pred regularized (Firing Power)')
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='seizure Threshold')
        ax.axvline(seizure_onset + int(diff.total_seconds()), label='seizure onset', c='r', alpha=.6)
        ax.axvline(seizure_onset + seizure_duration + int(diff.total_seconds()), label='seizure end', c='r', alpha=.6)
        ax.xaxis.set_major_formatter(formatter)

        if plot_alarms:
            for xv in asc_zc:
                plt.axvline(x=tm[xv], color='g', linestyle='dotted', label='alarm')

        legend_without_duplicate_labels(ax)

        if save_fig:
            plt.savefig(save_fig + '_pred.png', dpi=300, bbox_inches='tight')

        plt.show()

    n_missed = 0
    try:
        s_detect = next(x for x in asc_zc if tm[x] > seizure_onset + int(diff.total_seconds()))
        latency = tm[s_detect] - (seizure_onset + int(diff.total_seconds()))
        false_alarms = [x for x in asc_zc if x < s_detect]
    except StopIteration:
        print('seizure was not detected')
        latency = None
        n_missed += 1
        false_alarms = asc_zc

    # thresh = 48 = 4 minutes for a 5s-step (4*60/5=48)
    # 4min is chosen as time for heart rate recovery for people with epilepsy
    # doi S0165183800000904
    # doi 00207454.2012.683218
    # positive predictions within 4min of each other are attributed to the same event
    # multiple epoch-based FPs within this period are considered as 1 FP = 1 alarm
    # multiple epoch-based TP within this period are considered as 1 TP = 1 alarm

    return {'predictions': predictions[sz_0:sz_1+1],
            'regularized_predictions': reg_pred,
            'alarms_idx': asc_zc,
            'detection_latency': latency,
            'nb_false_alarms': len(false_alarms),
            'nb_missed_alarms': n_missed}


def predictions_per_record(test_dataset, record_row: int, model, scaled_X_test,
                           tau=6, threshold=0.85, step=5,
                           plot_alarms=False, show_plots=True, plot_features=False,
                           save_fig=None, home_path=home):
    """
    Generate predictions on the specified record given a trained model, and calculate performance metrics.
    Args:
        test_dataset : data with test seizures
        record_row   : record index in data
        model        : trained model to evaluate
        scaled_X_test: test features scaled using mean of train data
        tau          : window size for firing power regularization
        threshold    : threshold for firing power regularization
        step         : step that was used for moving window feature extraction
        plot_alarms  : bool, plot vertical lines when firing power average crosses the threshold
        show_plots   : bool, plot the features and predictions for 1h preictal of the selected seizure
        save_fig     : str, default=None, is path provided, save the plots as png.
        home_path    : str, home directory for the EDF and features
    Returns:
        dictionary with performances measures of the selected record.
        {
        OVLP: dictionary of TP, FP, FN, precision, recall and f1-score (event-based)
        Detection_latency: latency between seizure onset and start of true positive event
        TP_SZ_overlap: overlap between seizure and true positive event
        Time_in_warning: duration of false positive events
        percentage_tiw: (total time in warning)/(record duration)
        TP_duration: duration of true positive events
        FN_duration: duration of false negative events (missed seizures)
        FP_hours: hours of the day of the false alarms
        TP_hours: hours of the day of the true alarms
        FN_hours: hours of the day of the missed alarms/seizures
        regularized_predictions: array of regularized predictions representing high-risk alarms
        }
    """

    # X_test = test_dataset.iloc[:, 4:-1]
    # y_test = test_dataset.iloc[:, -1]
    predictions = model.predict(scaled_X_test)

    fs = 256
    # step = (test_dataset.iloc[1].timestamp - test_dataset.iloc[0].timestamp) / fs

    # NOTE: if choosing a different step, adjust assertion
    assert step == 5 or step == 3
    test_rec = test_dataset[test_dataset.Record_index == record_row]
    record = test_rec['Record_name'].unique()[0]
    p = test_rec['Patient_ID'].unique()[0]

    logger.info(f'\nPatient {p}, record {record_row}\n')
    """
    edf_path = ''
    for directory in os.listdir(data_path):
        if 'p' + str(p) in directory:
            edf_path = data_path + directory + '/' + record
            print(edf_path)

    info, edf_info, orig_units = mne.io.edf.edf._get_info(edf_path, stim_channel='auto',
                                                          eog=None, misc=None, exclude=(),
                                                          infer_types=True, preload=False)
    record_duration = edf_info['n_records']

    rec_dt = datetime.datetime.combine(edf_info['meas_date'].date(), datetime.datetime.min.time())
    rec_dt = rec_dt.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
    diff = edf_info['meas_date'] - rec_dt
    """
    # To avoid reading edf for header details
    logger.info(record)
    meas_date = datetime.datetime.strptime(records.Record_start_date[record_row], '%Y-%m-%d %H:%M:%S%z')

    record_duration = records.n_records[record_row]

    rec_dt = datetime.datetime.combine(meas_date.date(), datetime.datetime.min.time())
    rec_dt = rec_dt.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
    diff = meas_date - rec_dt
    # ##
    tm = test_rec['timestamp']
    tm = np.array(tm) / fs + int(diff.total_seconds())
    hrv_col = [x for x in test_rec.columns if 'HRV' in x]
    acm_col = [x for x in test_rec.columns if 'Gait' in x]
    df_acm = test_rec[acm_col]
    df_hrv = test_rec[hrv_col]

    rec_idx = test_rec.index.tolist()
    test_lst = test_dataset.index.tolist()
    rec_0 = test_lst.index(rec_idx[0])
    rec_1 = test_lst.index(rec_idx[-1])

    # Regularize prediction output with firing power method
    reg_pred = firing_power(predictions[rec_0:rec_1 + 1], tau=tau)  # , threshold=threshold)

    if save_fig:
        save_fig = save_fig + f'_p{p}_{record_row}_{record[:-4]}'  # + str(p) + '_' + str(record_row)
        # plot_ACM_features_record(tm, df_acm, records, record_row,
        #                          show=show_plots, save_fig=save_fig + '_ACM.png')
        # plot_tHRV_with_timestamp_record(tm, df_hrv, records, record_row,
        #                                 show=show_plots, save_fig=save_fig + '_HRV_time.png')
        # plot_nlHRV_with_timestamp_record(tm, df_hrv, records, record_row,
        #                                  show=show_plots, save_fig=save_fig + '_HRV_nonlinear.png')
    elif show_plots and plot_features:
        plot_ACM_features_record(tm, df_acm, records, record_row, show=show_plots)
        plot_tHRV_with_timestamp_record(tm, df_hrv, records, record_row, show=show_plots)
        plot_nlHRV_with_timestamp_record(tm, df_hrv, records, record_row, show=show_plots)
    else:
        pass

    Desc = records.Descriptions[record_row]
    Desc = Desc.replace("[", "").replace("]", "").replace("'", "").split(', ') if not (pd.isna(Desc)) else []
    Onsets = records.Onsets[record_row]
    Onsets = Onsets.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Onsets)) else []
    Onsets = [int(x) for x in Onsets]
    Durations = records.Durations[record_row]
    Durations = Durations.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Durations)) else []
    Durations = [int(x) for x in Durations]

    true_ann = np.array(test_rec.annotation)
    pred_ann = np.array(reg_pred)
    pred_ann = np.array([1 if x >= threshold else 0 for x in pred_ann])
    OVLP = ovlp(true_ann, pred_ann, step)

    # ########################################################################################################
    # DO NOT REMOVE COMMENTED LINES IN THIS SECTION
    # EXTRA PLOTS ALLOW VERIFYING DETAILS
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set(xlabel="Time (UTC)", title="True vs predicted labels for patient %a, record %s" % (p, record_row))
    # ax.plot(tm, test_rec['annotation'], c='g', label='y_true')
    ax.scatter(tm, predictions[rec_0:rec_1 + 1], label='y_pred', alpha=.3, linewidths=.3)
    # ax.plot(tm, reg_pred, marker='.', label='y_pred regularized (Firing Power)', alpha=.3)
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Alarm Threshold')

    pe = OVLP['pred_events']
    for e in pe:
        ev0 = test_rec.iloc[pe[e][0]]['timestamp'] / fs + int(diff.total_seconds())
        ev1 = test_rec.iloc[pe[e][-1]]['timestamp'] / fs + int(diff.total_seconds())
        if e in OVLP['pred_sz_events']:
            ax.axvspan(ev0, ev1, alpha=.5, color='green', label='True alarm')
        elif e in OVLP['pred_fp_events']:
            ax.axvspan(ev0, ev1, alpha=.5, color='orange', label='False alarm')

    # te = OVLP['true_events']
    # for t in te :
    #     sz0 = test_rec.iloc[te[t][0]]['timestamp']/fs+int(diff.total_seconds())
    #     sz1 = test_rec.iloc[te[t][-1]]['timestamp']/fs+int(diff.total_seconds())
    #     ax.axvspan(sz0, sz1, alpha=.5, color='cyan', label='true_events')

    for s in range(len(Desc)):
        sz_onset = Onsets[s] + int(diff.total_seconds())
        sz_end = Onsets[s] + Durations[s] + int(diff.total_seconds())
        if Desc[s] == 'FBTCS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='red', label='FBTCS')
        elif Desc[s] == 'FIAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='skyblue', label='FIAS')
        elif Desc[s] == 'FAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='grey', label='FAS')
        elif Desc[s] == 'FUAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='powderblue', label='FUAS')

    ax.xaxis.set_major_formatter(formatter)

    if plot_alarms:
        a_pred = np.array(reg_pred) - threshold
        asc_zc = np.where(np.diff(np.sign(a_pred)) > 0)[0]  # crossing threshold in an ascending way
        for xv in asc_zc:
            plt.axvline(x=tm[xv + 1], color='g', linestyle='dotted', label='alarm')
    # legend_without_duplicate_labels(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    if save_fig:
        plt.savefig(save_fig + '_pred.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    latency, overlap = [], []
    tiw, TP_duration, FN_duration = [], [], []
    FP_hours, TP_hours, FN_hours = [], [], []
    for i in OVLP['pred_sz_events']:
        l = []  # Detection latency between TP event and all available seizure on the record
        w = []  # Overlap between true seizure and detection event
        ev0 = test_rec.iloc[OVLP['pred_events'][i][0]]['timestamp'] / fs
        ev1 = test_rec.iloc[OVLP['pred_events'][i][-1]]['timestamp'] / fs
        for j in range(len(Onsets)):
            l.append(ev0 - Onsets[j])
            if (ev0 > Onsets[j]) and (ev0 < Onsets[j] + Durations[j]):
                # To avoid accounting overlap if seizure is detected slightly after seizure offset
                w.append(min(Onsets[j] + Durations[j], ev1) - ev0)
        l = np.abs(l)
        w = np.abs(w)
        # Detection latency
        latency.append(min(l))  # min is to find the correct seizure associated to the true positive event
        # Overlap between true seizure reference and correct predicted event
        overlap.extend(w)
        # Hours of the day when seizures are detected (does not account for missed seizures)
        tp_hour = event_hour_mtl_tz(ev0, rec_dt, diff)
        TP_hours.append(tp_hour)
        # True alarm duration
        TP_duration.append(ev1 - ev0)

    for i in OVLP['pred_fp_events']:
        ev0 = test_rec.iloc[OVLP['pred_events'][i][0]]['timestamp'] / fs
        ev1 = test_rec.iloc[OVLP['pred_events'][i][-1]]['timestamp'] / fs
        fp_hour = event_hour_mtl_tz(ev0, rec_dt, diff)
        tiw.append(ev1 - ev0)  # TiW Time in warning is duration of FP.
        FP_hours.append(fp_hour)  # Hour of the day of the false alarm

    for i in OVLP['missed_sz']:
        ev0 = test_rec.iloc[OVLP['true_events'][i][0]]['timestamp'] / fs
        ev1 = test_rec.iloc[OVLP['true_events'][i][-1]]['timestamp'] / fs
        fn_hour = event_hour_mtl_tz(ev0, rec_dt, diff)
        FN_duration.append(ev1 - ev0)  # Duration of missed seizure
        FN_hours.append(fn_hour)  # Hour of the day of the missed seizure

    # print('Detection latency', latency, '\t', np.mean(latency))
    # print('Time in warning', tiw, '\t', np.mean(tiw))

    percentage_tiw = np.sum(tiw) / record_duration * 100
    percentage_tiw = round(percentage_tiw, 4)
    # False alrm rate per day = nbr false alarms in record *24h / record duration
    FAR = len(tiw) * 24 * 3600 / record_duration
    FAR = round(FAR, 4)

    #  Sample-based metrics to estimate performance on records that contain no seizures.
    try:
        assert len(true_ann) == len(pred_ann)
        pres_rec, rec_rec, f1_rec, support_rec = precision_recall_fscore_support(true_ann, pred_ann,
                                                                                 zero_division=0)
        s = len(support_rec)
        msg = f"""Sample-based performance on record {record_row}:  \
        f1={100 * float('{:.4f}'.format(f1_rec[s-1]))}% \
        precision={100 * float('{:.4f}'.format(pres_rec[s-1]))}% \
        recall={100 * float('{:.4f}'.format(rec_rec[s-1]))}% \
        Tiw = {round(np.sum(tiw)/3600, 2)}h/{round(record_duration/3600,1)}h = {percentage_tiw}%"""
        logger.info(msg)
    except (AssertionError, ValueError) as error:
        logger.error(f'ERROR {error} len(true_ann)={len(true_ann)}, len(pred_ann)={len(pred_ann)}',
                     exc_info=True)

    return {'OVLP': OVLP, 'Detection_latency': latency,
            'TP_SZ_overlap': overlap, 'FAR': FAR,
            'Time_in_warning': tiw, 'percentage_tiw': percentage_tiw,
            'FP_hours': FP_hours, 'TP_hours': TP_hours, 'FN_hours': FN_hours,
            'TP_duration': TP_duration, 'FN_duration': FN_duration,
            'regularized_predictions': pred_ann}


def utc_to_mtl(utc_dt):
    """ Convert UTC timezone to Montreal timezone """
    mtl_tz = pytz.timezone('America/Montreal')
    conv_mtl_tz = utc_dt.replace(tzinfo=pytz.utc).astimezone(mtl_tz)
    return conv_mtl_tz


def event_hour_mtl_tz(tmp, dtm, diff):
    """ tmp : timestamp in seconds, from feature_df.iloc[row].timestamp/fs """
    ev_mtl_dt = utc_to_mtl(datetime.datetime.combine(dtm.date(),
                                                     datetime.time()) + diff + datetime.timedelta(seconds=tmp))
    return ev_mtl_dt.hour


# def split_data(dataset, val_size=.15, test_size=.15, random_state=7):
#     splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=random_state)
#     split = splitter.split(dataset, groups=dataset['Seizure_index'])
#     train_val_idx, test_idx = next(split)
#
#     train_val = dataset.iloc[train_val_idx]
#     test = dataset.iloc[test_idx]
#
#     # X_train_val = train_val.iloc[:, 4:-1]
#     # y_train_val = train_val.iloc[:, -1]
#
#     X_test = test.iloc[:, 4:-1]
#     y_test = test.iloc[:, -1]
#
#     splitter = GroupShuffleSplit(test_size=val_size, n_splits=5, random_state=random_state)
#     split = splitter.split(train_val, groups=train_val['Seizure_index'])
#     train_idx, val_idx = next(split)
#
#     sc = StandardScaler()
#     X_train_val = sc.fit_transform(train_val.iloc[:, 4:-1])
#
#     X_train = X_train_val[train_idx]
#     y_train = train_val.iloc[train_idx, -1]
#
#     X_val = X_train_val[val_idx]
#     y_val = train_val.iloc[val_idx, -1]
#
#     X_test = sc.transform(X_test)
#
#     train_set = train_val.iloc[train_idx]
#     val_set = train_val.iloc[val_idx]
#
#     # X_train = train.iloc[:, 4:-1]
#     # y_train = train.iloc[:, -1]
#     #
#     # sc = StandardScaler()
#     # X_train = sc.fit_transform(X_train)
#     # X_test = sc.transform(X_test)
#     #
#     # X_val = val.iloc[:, 4:-1]
#     # y_val = val.iloc[:, -1]
#
#     print('Number of seizures in full dataset', len(dataset.Seizure_index.unique()))
#     print('Number of seizures in train/validation set', len(train_val.Seizure_index.unique()),
#           '(', len(train_set.Seizure_index.unique()),
#           ',', len(val_set.Seizure_index.unique()), ')')
#     print('Number of seizures in test set', len(test.Seizure_index.unique()))
#     print()
#     print('Seizure IDs to be used in test', test.Seizure_index.unique())
#     print('Patients with selected test seizures', test.Patient_ID.unique())
#
#     return {'train_set': train_set,
#             'scaled_X_train': X_train, 'y_train': y_train,
#             'val_set': val_set,
#             'scaled_X_val': X_val, 'y_val': y_val,
#             'test_set': test,
#             'scaled_X_test': X_test, 'y_test': y_test}


# def train(X_train, y_train, params):
#     """
#     This function trains a linear support vector machine classifier,
#     using balanced class weight to counteract the unbalance in data,
#     regularization parameter C is 1.0
#
#     Args:
#         X_train: scaled training features
#         y_train: true labels in training
#         params : estimator parameters {'C': c_value, 'kernel': kernel}
#
#     Returns:
#         instance of fitted SVC, with balanced class weight
#
#     """
#
#     svm = SVC(class_weight='balanced')
#     svm.set_params(**params)
#     # linear_svc = LinearSVC(dual=False, class_weight='balanced')
#     # X_train = train_set.iloc[:,4:-1]
#     # y_train = train_set.iloc[:, -1]
#     svm.fit(X_train, y_train.ravel())
#     print(svm)
#     return svm


def ovlp(true_ann, predictions):
    """
    Any-overlap method for event-based evaluation of seizure detection.
    Args:
        true_ann: ground truth annotations
        predictions: binary output of the classifier
    Returns:
        dictionary of TP, FP, FN, precision, recall and f1-score.
    """
    # pred = predictions_per_seizure(split['test_set'], seizure_row=s, model=loaded_model,
    #                                scaled_X_test=split['scaled_X_test'],
    #                                tau=tau, threshold=threshold,
    #                                plot_alarms=True,
    #                                show_plots=True);
    #
    # true_ann = split['test_set'][split['test_set'].Seizure_index == s].annotation
    # pred_ann = pred['regularized_predictions']
    # pred_ann = [1 if x > threshold else 0 for x in pred_ann]
    indexes = [i for i, x in enumerate(predictions) if x == 1]
    # tm = np.array(split['test_set'][split['test_set'].Seizure_index == s].timestamp)

    events = dict(enumerate(grouper(indexes, thres=48), 1))
    # thresh = 48 = 4 minutes for a 5s-step (4*60/5=48)
    # 4min is chosen as time for heart rate recovery for people with epilepsy

    # doi S0165183800000904
    # doi 00207454.2012.683218

    # positive predictions within 4min of each other are attributed to the same event
    # multiple epoch-based FPs within this period are considered as 1 FP = 1 alarm
    # multiple epoch-based TP within this period are considered as 1 TP = 1 alarm
    # for i in events.keys():
    #     print(i, events[i])

    grps = []
    for i in indexes:
        grps.append(get_key(events, i))

    TP = 0
    FN = 1
    if np.sum(predictions*true_ann) > 0:
        TP = 1
        FN = 0
    FP = len(events.keys()) - TP

    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = np.nan
        print(f'  OVLP - - TP={TP}, FP={FP}, precision is set to nan')

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = np.nan
        print(f'  OVLP - - TP={TP}, FN={FN}, recall is set to nan')
    try:
        f1 = 2 * (precision*recall)/(precision+recall)
        print(f'  OVLP - - f1 {f1*100:.2f}%')
    except ZeroDivisionError:
        f1 = np.nan
        print(f'  OVLP - - precision={precision}, recall={recall}, f1 is set to nan')

    return {'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1': f1}


def ovlp(true_ann, predictions, step):
    """
        Any-overlap method for event-based evaluation of seizure detection.
        Args:
            true_ann: ground truth annotations
            predictions: binary output of the classifier
            step: step used for feature extraction. (in seconds)
        Returns:
            dictionary of event-based metrics:
            TP, FP, FN, precision, recall and f1-score.
            true_events: dictionary of reference seizure events containing seizure onset timestamps
            pred_events: dictionary of all events/alarms generated by the estimator
            pred_sz_events: index of true positive events from pred_events
            pred_fp_events: index of false positive events from pred_events
            detected_sz: index of correctly detected seizures from true_events
            missed_sz: index of missed seizures from true_events
        """
    true_indexes = np.array([i for i, x in enumerate(true_ann) if x == 1])
    pred_indexes = np.array([i for i, x in enumerate(predictions) if x == 1])
    th = int(4 * 60 / step)
    # NOTE: 4min is chosen as time for heart rate recovery for people with epilepsy
    # thresh = 48 = 4 minutes for a 5s-step (4*60/5=48)
    # thresh = 80 = 4 minutes for a 3s-step (4*60/3=80)
    # doi S0165183800000904
    # doi 00207454.2012.683218

    # seizures within 4min of each other are attributed to the same true event
    true_events = dict(enumerate(grouper(true_indexes, thres=th), 1))
    # positive predictions within 4min of each other are attributed to the same event
    pred_events = dict(enumerate(grouper(pred_indexes, thres=th), 1))

    sz_events = []
    detected_sz = []
    missed_sz = []
    for i in true_events:
        if np.sum(true_ann[true_events[i]] * predictions[true_events[i]]) > 0:  # Any-overlap
            for j in true_events[i]:
                if predictions[j] == 1:
                    sz_key = get_key(pred_events, j)
                    sz_events.append(sz_key)  # index of detected seizure among predicted events
                    detected_sz.append(i)  # index of detected seizure among all FBTCS events
                    # Detection latency on true alarms, in seconds.
                    # latency.append(true_events[i].index(j) * step)
                    # latency.append((pred_events[sz_key][0]-true_events[i][0]) * step)
                    # this formula for latency is incorrect in case of gaps in observations
                    # FIXED : latency is calculated outside this function
                    break
        else:
            missed_sz.append(i)

    # print()
    fp_events = list(pred_events.keys())
    for s in sz_events:
        fp_events.remove(s)
    TP = len(sz_events)
    FN = len(missed_sz)
    FP = len(fp_events)

    # TiW : Time in warning in seconds, calculated on false alarms only.
    # tiw = [len(pred_events[i]) * step for i in fp_events]
    # this formula is wrong if there's a gap in observations
    # or if an FP event is 2 smaller events grouped together, also doesn't consider the gap
    # FIXED: TiW is calculated outside this function

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        # recall = np.nan
        recall = 1
        logger.info('record has no seizures')
        logger.info(f'  OVLP - - TP={TP}, FN={FN}, recall is set to 1')

    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        if TP + FN == 0:  # record has no seizures
            logger.info('no false positives')
            precision = 1  # TP=FP=FN=0
            logger.info(f'  OVLP - - TP={TP}, FN={FN}, precision is set to 1')
        else:  # FN!=0  > missed seizure
            precision = 0  # record has seizure(s) but no alarm was launched, TP or FP
            logger.info(f'  OVLP - - TP={TP}, FP={FP}, precision is set to 0')

    try:
        f1 = 2 * (precision*recall)/(precision+recall)
        logger.info(f'  OVLP - - f1 {f1*100:.2f}%')
    except ZeroDivisionError:
        # f1 = np.nan
        f1 = 0
        logger.info(f'  OVLP - - precision={precision}, recall={recall}, f1 is set to 0')

    # FIXED
    # Consider the records that contain no seizures:
    # Ideally, TP=0, FP=0 and FN=0 -> precision=100, recall=100 and f1=100
    # if FP>0 , then recall=100 and precision=0

    return {'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1': f1,
            'true_events': true_events, 'pred_events': pred_events,
            'pred_sz_events': sz_events, 'pred_fp_events': fp_events,
            'detected_sz': detected_sz, 'missed_sz': missed_sz}


def calculate_metrics(model, test_set, scaled_X_test,
                      tau=6, threshold=.85, step=5,
                      show=True, home_path=home):
    """
    Estimate class of the input test set samples given a trained model,
    implement regularization using firing power method and evaluate performance.
    Args:
        model        : trained model for predictions
        test_set     : test or validation set to be used
        scaled_X_test: scaled testing features
        tau          : (regularization hyperparameter) window size for averaging firing power
        threshold    : (regularization hyperparameter) minimum fraction of "full" firing power to raise an alarm
        step         : step that was used for feature extraction
        show         : bool - if True, plots features and prediction per seizure in the given set
        home_path    : str, home directory for the EDF and features
    Returns:
        dict of metrics to evaluate the model
    """

    # X_test = test_set.iloc[:, 4:-1]
    y_test = test_set.iloc[:, -1]
    latencies = []
    TP_SZ_overlap = []
    # nb_FA, nb_MA = 0, 0
    # y_pred = []
    y_hat = []
    ovlp_precision, ovlp_recall, ovlp_f1 = [], [], []
    ovlp_FA, ovlp_MA = 0, 0
    FP_h, TP_h, FN_h = [], [], []
    TP_duration, FN_duration = [], []
    tiw, percentage_tiw = [], []
    FAR = []

    # for i in test_set.Seizure_index.unique():   # for predictions_per_seizure
    for i in test_set.Record_index.unique():
        # Predictions per seizure ______________________________________________________________________________________
        # pred = predictions_per_seizure(test_set, seizure_row=i, model=model,
        #                                scaled_X_test=scaled_X_test,
        #                                tau=tau, threshold=threshold,
        #                                plot_alarms=show, show_plots=show,
        #                                home_path=home_path);
        # latencies.append(pred['detection_latency'])  # sample-based
        # nb_FA += pred['nb_false_alarms']  # sample-based
        # nb_MA += pred['nb_missed_alarms']  # sample-based
        # # y_pred.extend(pred['predictions'])
        #
        # # # FP_s = firing_power(test_set[test_set.Seizure_index == i].annotation, tau=tau, threshold=threshold)
        # # # print('FP_s = firing_power(test_set[test_set.Seizure_index == i].annotation)')
        # FP_s = pred['regularized_predictions']
        # y_pred_reg.extend(FP_s)
        # # ______________________________
        # true_ann = test_set[test_set.Seizure_index == i].annotation
        # pred_ann = pred['regularized_predictions']
        # pred_ann = [1 if x > threshold else 0 for x in pred_ann]
        # OVLP = ovlp(true_ann, pred_ann)
        # ovlp_precision.append(OVLP['precision'])  # event-based
        # ovlp_recall.append(OVLP['recall'])  # event-based
        # ovlp_f1.append(OVLP['f1'])  # event-based
        # ovlp_FA += OVLP['FP']  # event-based
        # ovlp_MA += OVLP['FN']  # event-based
        # # ______________________________
        # ______________________________________________________________________________________________________________
        # Predictions per record _______________________________________________________________________________________
        pred = predictions_per_record(test_set, record_row=i, model=model,
                                      scaled_X_test=scaled_X_test,
                                      tau=tau, threshold=threshold, step=step,
                                      plot_alarms=show, show_plots=show,
                                      home_path=home_path)  # ;

        # Event-based metrics
        OVLP = pred['OVLP']
        ovlp_precision.append(OVLP['precision'])
        ovlp_recall.append(OVLP['recall'])
        ovlp_f1.append(OVLP['f1'])
        ovlp_FA += OVLP['FP']
        ovlp_MA += OVLP['FN']

        latencies.extend(pred['Detection_latency'])
        TP_SZ_overlap.extend(pred['TP_SZ_overlap'])
        FAR.append(pred['FAR'])
        FP_h.extend(pred['FP_hours'])
        TP_h.extend(pred['TP_hours'])
        FN_h.extend(pred['FN_hours'])
        tiw.extend(pred['Time_in_warning'])
        TP_duration.extend(pred['TP_duration'])
        FN_duration.extend(pred['FN_duration'])
        percentage_tiw.append(pred['percentage_tiw'])
        y_hat.extend(pred['regularized_predictions'])

    # L = [1 if x >= threshold else 0 for x in y_hat]
    # sample-based, on all test dataset
    try:
        assert len(y_test) == len(y_hat)
        pres_rg, rec_rg, f1_rg, support_rg = precision_recall_fscore_support(y_test, y_hat,
                                                                             zero_division=0)
        s = len(support_rg)
        f1_rg = 100 * float("{:.4f}".format(f1_rg[s - 1]))
        pres_rg = 100 * float("{:.4f}".format(pres_rg[s-1]))
        rec_rg = 100 * float("{:.4f}".format(rec_rg[s-1]))

        if model.__class__.__name__ in ['LogisticRegression', 'SVC', 'XGBClassifier']:
            # y_pred_score = model.decision_function(scaled_X_test)
            # roc = roc_auc_score(y_test, y_pred_score)  # sample-based
            y_pred_score = model.predict_proba(scaled_X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred_score)  # sample-based
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_score)  # sample-based
        else:
            roc = np.nan
            fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])

    except (AssertionError, ValueError) as error:
        logger.error(f'ERROR {error} len(true_ann)={len(y_test)}, len(pred_ann)={len(y_hat)}',
                     exc_info=True)
        f1_rg = np.nan
        pres_rg = np.nan
        rec_rg = np.nan
        roc = np.nan
        fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])

    # latencies = np.array(latencies, dtype=np.float64)

    # Return with predictions_per_seizure---------------------------------------------------#
    # return {'avg_latency': np.nanmean(latencies),                                         #
    #         'total_false_alarms': nb_FA,                                                  #
    #         'total_missed_alarms': nb_MA,                                                 #
    #         'f1_score_regularized': 100 * float("{:.4f}".format(f1_rg[1])),               #
    #         'precision_regularized': 100 * float("{:.4f}".format(pres_rg[1])),            #
    #         'recall_regularized': 100 * float("{:.4f}".format(rec_rg[1])),                #
    #         'roc_auc_score': 100 * float("{:.4f}".format(roc)),                           #
    #         'ovlp_precision': 100 * float("{:.4f}".format(np.nanmean(ovlp_precision))),   #
    #         'ovlp_recall': 100 * float("{:.4f}".format(np.nanmean(ovlp_recall))),         #
    #         'ovlp_f1': 100 * float("{:.4f}".format(np.nanmean(ovlp_f1))),                 #
    #         'ovlp_FA': ovlp_FA, 'ovlp_MA': ovlp_MA}                                       #
    # --------------------------------------------------------------------------------------#

    return {'f1_score_regularized': f1_rg,  # sample-based
            'precision_regularized': pres_rg,  # sample-based
            'recall_regularized': rec_rg,  # sample-based
            'roc_auc_score': float("{:.4f}".format(roc)),  # sample-based
            'roc_fpr': fpr, 'roc_tpr': tpr, 'roc_thresholds': thresholds,  # sample-based
            'ovlp_precision': 100 * float("{:.4f}".format(np.nanmean(ovlp_precision))),  # event-based
            'ovlp_recall': 100 * float("{:.4f}".format(np.nanmean(ovlp_recall))),  # event-based
            'ovlp_f1': 100 * float("{:.4f}".format(np.nanmean(ovlp_f1))),  # event-based
            'ovlp_FA': ovlp_FA, 'ovlp_MA': ovlp_MA,  # event-based
            'latencies': latencies,  # event-based
            'TP_SZ_overlap': TP_SZ_overlap,  # event-based
            'FAR': FAR,  # event-based
            'Time_in_warning': tiw, 'percentage_tiw': percentage_tiw,  # event-based
            'TP_duration': TP_duration, 'FN_duration': FN_duration,  # event-based
            'FP_hours': FP_h, 'TP_hours': TP_h, 'FN_hours': FN_h  # event-based
            }


def plot_pr_curve(test_y, model_probs):
    """
    Args:
        test_y: true labels
        model_probs: probability estimates for all classes predicted by the model

    Returns:
        plots no-skill and model precision-recall curves
    """
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(test_y[test_y == 1]) / len(test_y)

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label='Logistic')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def split_patients_per_seizures(data, k=5):
    """
    Split data according to patient_ID to ensure no dipping of same patient data between train and test sets.
    The groups of patients in the splits have the same number of seizures (if possible)
    Args:
        data: features dataframe containing patient_ID
        data is assumed to have columns Patient_ID and Seizure_index.
        k: number of folds, default 5 folds.
    Returns:
        dict: dictionary containing a list of patient_ID to include in test set in each fold.
    """
    nbr_per_patient = []
    patient_seizures = {}
    for p in data.Patient_ID.unique():
        # print(p, data[data.Patient_ID == p].Seizure_index.unique())
        patient_seizures[p] = list(data[data.Patient_ID == p].Seizure_index.unique())
        nbr_per_patient.append(len(data[data.Patient_ID == p].Seizure_index.unique()))

    x = list(patient_seizures.keys())
    shuffle(x)
    folds = {}
    f = 1
    try:
        sz_per_fold = int(len(data.Seizure_index.unique()) / k)
        assert sz_per_fold > 2  # some patients have 3 seizures.
    except AssertionError:
        print("To ensure no dipping, minimum seizures per fold is 3, k set to 5 folds.")
        k = 5
        sz_per_fold = int(len(data.Seizure_index.unique()) / k)

    print(f'{sz_per_fold} seizures per fold for {k} folds.')

    while f < k + 1:
        n_sz = len(patient_seizures[x[0]])
        grp = [x.pop(0)]
        i = 0
        while n_sz < sz_per_fold:
            try:
                if n_sz + len(patient_seizures[x[i]]) <= sz_per_fold:
                    n_sz += len(patient_seizures[x[i]])
                    grp.append(x.pop(i))
                else:
                    i += 1
            except IndexError:
                # print('\nLast 2 left have 3 seizures each. Reinitializing and restart shuffle...')
                x = list(patient_seizures.keys())
                shuffle(x)
                folds = {}
                f = 0
        if f == 0:
            f += 1
        else:
            folds['fold' + str(f)] = {'test': grp, 'train': list(set(data.Patient_ID.unique()) - set(grp))}
            f += 1

    return folds


def split_patients(patients_list, k):
    """
    Split a given list of patients into k groups of patients for train/test split folds.
    This split does not account for number of seizures per patient.
    Args:
        patients_list: list of patient IDs.
        k: int
         number of folds.
    Returns:
        dictionary of folds, containing train and test patients shuffled groups.
    """
    s = list(patients_list)
    shuffle(s)
    folds = {}
    s = [s[i::k] for i in range(k)]
    for i in range(k):
        folds[f'fold{i+1}'] = {'test': s[i], 'train': sum(s[0:i]+s[i+1:], [])}
    return folds


def grouper(iterable, thres=10):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= thres:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def get_key(my_dict, val):
    """ Return key of a value from a dictionary """
    for key, value in my_dict.items():
        if val in value:
            return key


def load_data(path, win_size, step):
    """
    Load `parquet` dataset according to selected window size and step that were used for feature extraction.
    Args:
        path: path to features directory
        win_size: original window size used for feature extraction {10, 15, 20, 25}
        step: original step used for sliding window feature extraction {3, 5}
    Returns:
        dataframe containing data
    """

    this_directory = path + f'/extracted_using_win{win_size}_step{step}'
    try:
        assert (win_size in {10, 15, 20, 25, 30} and step in {3, 5})
        # data = pd.read_csv(
        #     this_directory+'/features_dataset_supervised_win'+str(win_size)+'_step'+str(step)+'.csv'
        # )
        data = pd.read_parquet(
            this_directory + f'/_features_FBTCS_patients_supervised_win{win_size}_step{step}_.parquet'
        )
        # data = pd.read_parquet(
        #     this_directory + f'/_smallCVtest_features_FBTCS_patients_supervised_win{win_size}_step{step}_.parquet'
        # )
    except AssertionError:
        logger.error('Verify win_size or step', exc_info=True)
        data = pd.DataFrame()
    return data


def get_train_test_splits(patients_df, k: int):
    """
    Split dataframe into train and test split folds according to patient_ID, given K folds.
    Args:
        patients_df: dataframe of one column containing Patients_ID, extracted from original features dataframe.
        k: number of folds
    Returns:
        List of k tuples containing train and test indexes of the dataframe, for k folds.
    """
    idx = []
    patient_list = patients_df.Patient_ID.unique()

    splits = split_patients(patient_list, k=k)

    for i in range(k):
        # train_df = data[data['Patient_ID'].isin(splits['fold'+str(i+1)]['train'])]
        # test_df = data[data['Patient_ID'].isin(splits['fold'+str(i+1)]['test'])]
        # train_idx = train_df.index
        # test_idx = test_df.index
        train_idx = np.array(patients_df.loc[patients_df['Patient_ID'].isin(splits[f'fold{i+1}']['train'])].index)
        test_idx = np.array(patients_df.loc[patients_df['Patient_ID'].isin(splits[f'fold{i+1}']['test'])].index)
        idx.append((train_idx, test_idx))
    del train_idx, test_idx
    return idx


def fit_and_score(model, hp, data, train_idx, test_idx,
                  home_path=home):
    """Fit a model on training data and compute its score on test data.
    Args:
        model : scikit-learn estimator (will not be modified)
          the estimator to be evaluated
        hp : tuple
         Combination of hyperparameters to use when fitting the model.
        data : dataframe
          Subset dataframe of the original data, after outer split.
        train_idx : sequence of ints
          the indices of training samples (row indices of X)
        test_idx : sequence of ints
          the indices of testing samples
        home_path : str
          Home directory for the EDF and features
    Returns:
      dictionary of the performance metrics on test data
    """

    train_set = data.loc[train_idx]
    test_set = data.loc[test_idx]
    # In inner fold, test means validation set in the corresponding CV fold.

    X_train = train_set.iloc[:, 4:-1]
    y_train = train_set.iloc[:, -1]
    X_test = test_set.iloc[:, 4:-1]
    # y_test = test_set.iloc[:, -1]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = clone(model)
    if model.__class__.__name__ == 'SVC':
        params = {'kernel': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'LogisticRegression':
        params = {'solver': hp[0], 'C': hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        params = {'splitter': hp[0], 'criterion': hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        params = {'algorithm': hp[0], 'n_neighbors': hp[1]}
    elif model.__class__.__name__ == 'XGBClassifier':
        params = {'max_depth': hp[0], 'min_child_weight': hp[1],
                  'scale_pos_weight': 3000, 'max_delta_step': 1,
                  'learning_rate': 0.1, 'gamma': 0.1, 'booster': 'gbtree'}
        # NOTE: The ratio neg/pos can be recalculate in each fold and used for scale_pos_weight 
        # This could be optimized, but I doubt this will have a significant impact on results
    #   testing with 1 hour preictal, ratio was 33 sneg : 1spos
    #   testing on all data, ratio is 5696978 sneg : 1949 spos =(approx.) 2923
    else:
        params = {}

    model.set_params(**params)
    model.fit(X_train, y_train)
    # predictions = model.predict(X[test_idx])
    # score = score_fun(y[test_idx], predictions)

    metrics = calculate_metrics(model, test_set, scaled_X_test=X_test,
                                tau=hp[4], threshold=hp[5], step=hp[3],
                                show=False,
                                home_path=home_path)

    # score = metrics['ovlp_f1']
    score = metrics['f1_score_regularized']

    logger.info(f"\n\tInner CV loop: fit and evaluate one model; f1-score={score}%, ROC={metrics['roc_auc_score']}")
    if model.__class__.__name__ == 'SVC':
        logger.info(f'kernel:{hp[0]}, C:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]}')
    elif model.__class__.__name__ == 'LogisticRegression':
        logger.info(f'solver:{hp[0]}, C:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]}')
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        logger.info(f'splitter:{hp[0]}, criterion:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]},\
        threshold:{hp[5]}')
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        logger.info(f'algorithm:{hp[0]}, n_neighbors:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]},\
        threshold:{hp[5]}')
    elif model.__class__.__name__ == 'XGBClassifier':
        logger.info(f'max_depth :{hp[0]}, min_child_weight :{hp[1]}, win_size:{hp[2]},\
        step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]}')

    return metrics


def grid_search(model, hyperparams, data, inner_k: int, outer_fold_idx,
                home_path=home):
    """
    Exhaustive grid search over hyperparameter combinations for the specified estimator.
    Used for the inner loop of a nested cross-validation.
    The best hyperparameters selected after the cross validation are used to refit the estimator.
    Args:
        model : scikit-learn estimator
          The base estimator, copies of which are trained and evaluated. `model` itself is not modified.
        hyperparams : list[tuple]
          list of possible combinations of the hyperparameters grid.
        data : dataframe
          Subset dataframe of the original data (train/val set), after outer split.
        inner_k : int
          number of inner cross-validation folds
        outer_fold_idx : int
          Index of fold in outer cross-validation loop. Only for reference in log output.
        home_path : str
          Path to the home directory. Should be modified if working on server.
    Returns:
        best_model : scikit-learn estimator
          A copy of `model`, fitted on the whole `(X, y)` data, with the best (estimated) hyperparameter.
    """

    # X: numpy array of shape(n_samples, n_features) the design matrix
    # y: numpy array of shape(n_samples, n_outputs) or (n_samples,) the target vector
    # X and y are from train_val set after split for outer folds.

    inner_cv_results = pd.DataFrame(columns=['fold', 'train_Patient_ID',
                                             'max_depth', 'min_child_weight',
                                             'win_size', 'step',
                                             'tau', 'threshold',
                                             'avg_latency',
                                             'total_false_alarms', 'total_missed_alarms',
                                             'f1_score_regularized', 'f1_score_ovlp',
                                             'roc_auc',
                                             'precision_ovlp', 'recall_ovlp',
                                             'avg_TP_SZ_overlap', 'avg_far',
                                             'avg_tiw', 'avg_percentage_tiw'])

    win_size = hyperparams[0][2]
    step = hyperparams[0][3]
    clf = model.__class__.__name__
    directory = home_path + f'/Oumayma/Hexoskin/NestedCV/FBTCS_patients/{clf}/win{win_size}_step{step}/'

    all_scores = []
    FA_for_this_hp = []
    MA_for_this_hp = []
    precision_for_this_hp = []
    recall_for_this_hp = []
    f1_ovlp_for_this_hp = []
    roc_auc_for_this_hp = []
    latencies_for_this_hp = []
    TP_SZ_overlap_for_this_hp = []
    far_for_this_hp = []
    tiw_for_this_hp = []
    percentage_tiw_for_this_hp = []
    #######
    for j, hp in enumerate(hyperparams):
        logger.info(f"\n  Grid search: evaluate hyperparameters = \
        solver:{hp[0]}, C:{hp[1]}, win_size:{hp[2]}, step:{hp[3]}, tau:{hp[4]}, threshold:{hp[5]} ")

        scores_for_this_hp = []
        for train_idx, val_idx in get_train_test_splits(data[['Patient_ID']], inner_k):
            metrics = fit_and_score(model, hp, data,
                                    train_idx, val_idx,
                                    home_path=home_path)

            score = metrics['f1_score_regularized']  # sample-based
            scores_for_this_hp.append(score)
            # #######################################################################
            # latencies_for_this_hp.append(metrics['avg_latency'])  # sample-based
            # FA_for_this_hp.append(metrics['total_false_alarms'])  # sample-based
            # MA_for_this_hp.append(metrics['total_missed_alarms'])  # sample-based
            # precision_for_this_hp.append(metrics['ovlp_precision'])  # event-based
            # recall_for_this_hp.append(metrics['ovlp_recall'])  # event-based
            # f1_ovlp_for_this_hp.append(metrics['ovlp_f1'])  # event-based
            # #######################################################################

            # Event-based metrics
            FA_for_this_hp.append(metrics['ovlp_FA'])
            MA_for_this_hp.append(metrics['ovlp_MA'])
            precision_for_this_hp.append(metrics['ovlp_precision'])
            recall_for_this_hp.append(metrics['ovlp_recall'])
            f1_ovlp_for_this_hp.append(metrics['ovlp_f1'])
            roc_auc_for_this_hp.append(metrics['roc_auc_score'])

            latencies_for_this_hp.append(np.nanmean(np.array(metrics['latencies'])))
            TP_SZ_overlap_for_this_hp.append(np.nanmean(np.array(metrics['TP_SZ_overlap'])))
            far_for_this_hp.append(np.nanmean(np.array(metrics['FAR'])))
            tiw_for_this_hp.append(np.nanmean(np.array(metrics['Time_in_warning'])))
            percentage_tiw_for_this_hp.append(np.nanmean(np.array(metrics['percentage_tiw'])))
            ###########

        # NOTE: scoring used for hyperparameter tuning is f1-score calculated as sample-based.
        # DO NOT use regularized f1-score to optimize hyperparameter search
        all_scores.append(np.nanmean(scores_for_this_hp))

        #########
        inner_cv_results.at[j, 'fold'] = outer_fold_idx + 1
        inner_cv_results.at[j, 'train_Patient_ID'] = list(data.Patient_ID.unique())
        inner_cv_results.at[j, 'max_depth'] = hp[0]
        inner_cv_results.at[j, 'min_child_weight'] = hp[1]
        inner_cv_results.at[j, 'win_size'] = hp[2]
        inner_cv_results.at[j, 'step'] = hp[3]
        inner_cv_results.at[j, 'tau'] = hp[4]
        inner_cv_results.at[j, 'threshold'] = hp[5]
        inner_cv_results.at[j, 'avg_latency'] = np.nanmean(latencies_for_this_hp)
        inner_cv_results.at[j, 'total_false_alarms'] = np.nanmean(FA_for_this_hp)
        inner_cv_results.at[j, 'total_missed_alarms'] = np.nanmean(MA_for_this_hp)
        inner_cv_results.at[j, 'f1_score_regularized'] = np.nanmean(scores_for_this_hp)
        inner_cv_results.at[j, 'f1_score_ovlp'] = np.nanmean(f1_ovlp_for_this_hp)
        inner_cv_results.at[j, 'roc_auc'] = np.nanmean(roc_auc_for_this_hp)
        inner_cv_results.at[j, 'precision_ovlp'] = np.nanmean(precision_for_this_hp)
        inner_cv_results.at[j, 'recall_ovlp'] = np.nanmean(recall_for_this_hp)
        inner_cv_results.at[j, 'avg_TP_SZ_overlap'] = np.nanmean(TP_SZ_overlap_for_this_hp)
        inner_cv_results.at[j, 'avg_far'] = np.nanmean(far_for_this_hp)
        inner_cv_results.at[j, 'avg_tiw'] = np.nanmean(tiw_for_this_hp)
        inner_cv_results.at[j, 'avg_percentage_tiw'] = np.nanmean(percentage_tiw_for_this_hp)
        #########

    txt = directory + f'fold_{str(outer_fold_idx + 1)}_{clf}_grid_search_results.csv'
    inner_cv_results.to_csv(txt, index=False)

    # refit the model on the whole data using the best selected hyperparameter,
    # and return the fitted model.

    best_hp = hyperparams[np.argmax(all_scores)]
    # old_todo: Formula for choosing best score based on multiple metrics to be inserted in all_scores
    logger.info(f'Outer fold {outer_fold_idx+1} grid search finished')
    logger.info(f'\t ** Grid search: keep best hyperparameters combination = {best_hp} **')
    logger.info(f'\t ** Highest f1-score (regularized) from the grid search (on validation data) is {np.max(all_scores)}')
    # `clone` is to work with a copy of `model` instead of modifying the argument itself
    best_model = clone(model)

    # NOTE : the tuning of win_size and step used for feature extraction is complicated
    #  and should be done externally, in different Nested CV iterations:
    #  if we tune these 2 hyperparams, it means a different data set is to be loaded in each combination
    #  Consequently, if we split data to train/test in an outer loop, then tune the hyperparams in inner cv
    #  depending on the best hyperparameter selected, we have to reload the corresponding dataset
    #  this will certainly introduce a discrepancy in validation and testing process.
    #  Conclusion: Tuning of hyperparameters win_size and step should be done separately in an exterior loop .

    X = data.iloc[:, 4:-1]
    y = data.iloc[:, -1]

    sc = StandardScaler()
    X = sc.fit_transform(X)

    if model.__class__.__name__ == 'SVC':
        params = {'kernel': best_hp[0], 'C': best_hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'LogisticRegression':
        params = {'solver': best_hp[0], 'C': best_hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        params = {'splitter': best_hp[0], 'criterion': best_hp[1], 'class_weight': 'balanced'}
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        params = {'algorithm': best_hp[0], 'n_neighbors': best_hp[1]}
    elif model.__class__.__name__ == 'XGBClassifier':
        params = {'max_depth': best_hp[0], 'min_child_weight': best_hp[1],
                  'scale_pos_weight': 3000, 'max_delta_step': 1,
                  'learning_rate': 0.1, 'gamma': 0.1, 'booster': 'gbtree'}
    # NOTE: Since our dataset is imbalanced with 1h non-seizure data for approx. 2min of seizure data
    # we empirically set the XGBoost parameter scale_pos_weight to 3000 to control the balance of classes
    # Typical value to consider is sum(negative instances) / sum(positive instances)
    # Other parameters are set empirically to reduce overfitting and avoid underfiting
    # You could recalibrate scale_pos_weight in each fold

    else:
        params = {}
    best_model.set_params(**params)
    best_model.fit(X, y)

    return {'best_model': best_model, 'best_hyperparam': best_hp, 'scaler': sc}


def cross_validate(model, hyperparams, data, k, inner_k,
                   home_path=home):
    """Get CV score with an inner CV loop to select hyperparameters.

    Args:
        model : scikit-learn estimator, for example `LogisticRegression()`
          The base model to fit and evaluate. `model` itself is not modified.
        hyperparams : list[tuple]
          list of possible combinations of the hyperparameters grid.
        data : dataframe
          Subset dataframe of the original data, after outer split.
        k : int
          the number of splits for the k-fold cross-validation.
        inner_k : int
          the number of splits for the nested cross-validation (hyperparameter
          selection).
        home_path : str
          Path to the home directory. Should be modified if working on server.
    Returns:
        scores : list[float]
           The scores obtained for each of the cross-validation folds
    """

    df_results = pd.DataFrame(columns=['fold', 'train_Patient_ID', 'test_Patient_ID', 'model',
                                       'max_depth', 'min_child_weight', 'win_size', 'step',
                                       'tau', 'threshold',
                                       'avg_latency', 'total_false_alarms', 'total_missed_alarms',
                                       'f1_score_ovlp', 'precision_ovlp', 'recall_ovlp',
                                       'f1_score_regularized', 'roc_auc_score',
                                       'latencies', 'TP_SZ_overlap', 'FAR_per_day',
                                       'Time_in_warning', 'percentage_tiw',
                                       'TP_duration', 'FN_duration',
                                       'FP_hours', 'TP_hours', 'FN_hours'])
    df_roc_curves = pd.DataFrame(columns=['fold', 'roc_fpr', 'roc_tpr', 'roc_thresholds'])

    clf = model.__class__.__name__

    win_size = hyperparams[0][2]
    step = hyperparams[0][3]
    directory = home_path + f'/Oumayma/Hexoskin/NestedCV/FBTCS_patients/{clf}/win{win_size}_step{step}/'
    all_scores = []

    for i, (train_idx, test_idx) in enumerate(get_train_test_splits(data[['Patient_ID']], k)):
        df_roc = pd.DataFrame(columns=['fold', 'roc_fpr', 'roc_tpr', 'roc_thresholds'])
        logger.info(f"\nOuter CV loop: fold {i+1}")

        train_val_set = data.iloc[train_idx]
        test_set = data.iloc[test_idx]

        X_test = test_set.iloc[:, 4:-1]
        # y_test = test_set.iloc[:, -1]

        gridsearch_per_fold = grid_search(model, hyperparams, train_val_set, inner_k,
                                          outer_fold_idx=i,
                                          home_path=home_path)
        best_model = gridsearch_per_fold['best_model']
        best_hp = gridsearch_per_fold['best_hyperparam']
        scaler = gridsearch_per_fold['scaler']

        model_name = f'fold_{i+1}_{clf}_{best_hp[0]}_{best_hp[1]}_win{best_hp[2]}_step{best_hp[3]}.sav'
        filename = directory + model_name
        pickle.dump(best_model, open(filename, 'wb'))
        # best_model.save_model(filename)

        X_test = scaler.transform(X_test)
        
        # Make predictions and calculate performance metrics
        metrics = calculate_metrics(best_model, test_set, scaled_X_test=X_test,
                                    tau=best_hp[4], threshold=best_hp[5], step=best_hp[3],
                                    show=False)
        score = metrics['ovlp_f1']

        df_results.at[i, 'fold'] = i+1
        df_results.at[i, 'train_Patient_ID'] = list(train_val_set.Patient_ID.unique())
        df_results.at[i, 'test_Patient_ID'] = list(test_set.Patient_ID.unique())
        df_results.at[i, 'model'] = model_name
        df_results.at[i, 'max_depth'] = best_hp[0]
        df_results.at[i, 'min_child_weight'] = best_hp[1]
        df_results.at[i, 'win_size'] = best_hp[2]
        df_results.at[i, 'step'] = best_hp[3]
        df_results.at[i, 'tau'] = best_hp[4]
        df_results.at[i, 'threshold'] = best_hp[5]
        df_results.at[i, 'avg_latency'] = np.nanmean(np.array(metrics['latencies']))
        df_results.at[i, 'total_false_alarms'] = metrics['ovlp_FA']
        df_results.at[i, 'total_missed_alarms'] = metrics['ovlp_MA']
        df_results.at[i, 'f1_score_ovlp'] = score
        df_results.at[i, 'precision_ovlp'] = metrics['ovlp_precision']
        df_results.at[i, 'recall_ovlp'] = metrics['ovlp_recall']
        df_results.at[i, 'f1_score_regularized'] = metrics['f1_score_regularized']
        df_results.at[i, 'roc_auc_score'] = metrics['roc_auc_score']
        df_results.at[i, 'latencies'] = metrics['latencies']
        df_results.at[i, 'TP_SZ_overlap'] = metrics['TP_SZ_overlap']
        df_results.at[i, 'FAR_per_day'] = metrics['FAR']
        df_results.at[i, 'Time_in_warning'] = metrics['Time_in_warning']
        df_results.at[i, 'percentage_tiw'] = metrics['percentage_tiw']
        df_results.at[i, 'TP_duration'] = metrics['TP_duration']
        df_results.at[i, 'FN_duration'] = metrics['FN_duration']
        df_results.at[i, 'FP_hours'] = metrics['FP_hours']
        df_results.at[i, 'TP_hours'] = metrics['TP_hours']
        df_results.at[i, 'FN_hours'] = metrics['FN_hours']

        df_roc['fold'] = np.ones(len(metrics['roc_fpr'])) * (i+1)
        df_roc['roc_fpr'] = metrics['roc_fpr']
        df_roc['roc_tpr'] = metrics['roc_tpr']
        df_roc['roc_thresholds'] = metrics['roc_thresholds']

        logger.info(f"Outer CV loop: finished fold {i+1}, f1-score={score}%, ROC-AUC={metrics['roc_auc_score']}")
        all_scores.append(score)
        df_roc_curves = pd.concat([df_roc_curves, df_roc])

    df_results.to_csv(directory + 'nested_cross_validation_results.csv', index=False)
    df_roc_curves.to_csv(directory + 'nested_cv_results_ROC_curves.csv', index=False)

    return all_scores


def load_specific_data(train_patient_ID, test_patient_ID, data):
    train_idx = np.array(data.loc[data['Patient_ID'].isin(train_patient_ID)].index)
    test_idx = np.array(data.loc[data['Patient_ID'].isin(test_patient_ID)].index)

    train_set = data.loc[train_idx]
    test_set = data.loc[test_idx]

    X_train = train_set.iloc[:, 4:-1]
    # y_train = train_set.iloc[:, -1]
    X_test = test_set.iloc[:, 4:-1]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return test_set, X_test


def load_model(results, model, row, directory):
    # df = results.parse(model)
    df = results
    x = df.iloc[row]
    path = f'win{x.win_size}_step{x.step}'
    curr_dir = '/Features_FBTCS_patients/QRS_003/'
    parquet_file = curr_dir + f'extracted_using_{path}/_features_FBTCS_patients_supervised_{path}_.parquet'

    data = pd.read_parquet(directory + parquet_file)
    # data = pd.read_csv(folder + csv_file)
    filename = directory + f'/NestedCV/FBTCS_patients/{model}/{path}/QRS_003/'+x.model.replace(",", ".")
    loaded_model = pickle.load(open(filename, 'rb'))
    # if model == 'XGBClassifier':
    #     loaded_model = xgb.XGBClassifier()
    #     loaded_model.load_model(filename)

    train_ID = ast.literal_eval(x.train_Patient_ID)
    test_ID = ast.literal_eval(x.test_Patient_ID)
    test_set, scaled_X_test = load_specific_data(train_ID, test_ID, data)

    return loaded_model, test_set, scaled_X_test, x.tau, x.threshold



# -*- coding: utf-8 -*-
"""
Module for extracting 3D-accelerometry (ACC) features in temporal and spectral domains

@author: Oumayma Gharbi
"""

from utils import *


def normalize(v):
    """Normalization function"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def ACM_features(a, ax, ay, az, fs, start, end, win_size, step, plot=0):
    """ Computes features on acceleration data


    Parameters:
    ------------
    acm : list, shape = [n_samples,]
           Acceleration data = sqrt(x**2 + y**2 + z**2)

    fs : Sampling Frequency

    start : start of desired period to be analyzed in record, in seconds, used to synchronize with time on data
    end : end of desired period to be analyzed in record, in seconds

    win_size: size of extraction window in seconds

    step: Number of points to overlap between segments.

    plot : int, 0|1
          Setting plot to 1 creates a matplotlib figure showing features extracted
          from acceleration data and the histogram of frequencies in the signal.

    Returns:
    ---------
    ACM_feats : dict

    """

    timestamp = []

    ####################
    RMS, RMS_X, RMS_Y, RMS_Z = [], [], [], []
    ZCR, ZCR_X, ZCR_Y, ZCR_Z = [], [], [], []
    Energy, Energy_X, Energy_Y, Energy_Z = [], [], [], []
    MaxFFT, MaxFFT_X, MaxFFT_Y, MaxFFT_Z = [], [], [], []
    Gait_Min, Gait_Min_X, Gait_Min_Y, Gait_Min_Z = [], [], [], []
    Gait_STD, Gait_STD_X, Gait_STD_Y, Gait_STD_Z = [], [], [], []
    Gait_Mean, Gait_Mean_X, Gait_Mean_Y, Gait_Mean_Z = [], [], [], []
    Max_Value, Max_Value_X, Max_Value_Y, Max_Value_Z = [], [], [], []
    Gait_Norm, Gait_Norm_X, Gait_Norm_Y, Gait_Norm_Z = [], [], [], []
    Gait_Norm1, Gait_Norm1_X, Gait_Norm1_Y, Gait_Norm1_Z = [], [], [], []
    Diff_max_min, Diff_max_min_X, Diff_max_min_Y, Diff_max_min_Z = [], [], [], []
    Gait_Variance, Gait_Variance_X, Gait_Variance_Y, Gait_Variance_Z = [], [], [], []
    Gait_Skewness, Gait_Skewness_X, Gait_Skewness_Y, Gait_Skewness_Z = [], [], [], []
    Gait_Kurtosis, Gait_Kurtosis_X, Gait_Kurtosis_Y, Gait_Kurtosis_Z = [], [], [], []

    Gait_SD1 = []
    Gait_SD2 = []
    Gait_Ratio = []
    Corr_X_Y = []
    Corr_X_Z = []
    Corr_Y_Z = []

    # chunklist_a = list(slidingWindow(a, win_size, step))
    # chunklist_ax = list(slidingWindow(ax, win_size, step))
    # chunklist_ay = list(slidingWindow(ay, win_size, step))
    # chunklist_az = list(slidingWindow(az, win_size, step))

    # for sublist in chunklist:
    # for Gait, GaitX, GaitY, GaitZ in zip(chunklist_a, chunklist_ax, chunklist_ay, chunklist_az):
    for i in range(start*fs, end*fs, step * fs):

        Gait = a[i:i + win_size * fs]
        GaitX = ax[i:i + win_size * fs]
        GaitY = ay[i:i + win_size * fs]
        GaitZ = az[i:i + win_size * fs]

        timestamp.append(i + win_size * fs)
        # Gait
        Max_Value.append(Gait.max())
        Gait_Min.append(Gait.min())
        Diff_max_min.append(Gait.max() - Gait.min())
        Gait_STD.append(statistics.stdev(Gait))
        Gait_Mean.append(Gait.mean())
        Gait_Variance.append(Gait.var())
        Gait_Norm.append(LA.norm(Gait))
        Gait_Norm1.append(LA.norm(Gait, 1))
        MaxFFT.append(max(abs(fft(Gait))))
        Gait_Skewness.append(skew(Gait))
        Gait_Kurtosis.append(kurtosis(Gait))
        RMS.append(np.sqrt(mean(square(Gait))))  # rms
        Energy.append(np.sum(square(Gait)) * 1000)
        ZCR.append(librosa.feature.zero_crossing_rate(Gait)[0, 0])

        diff = np.diff(Gait)
        sd = np.std(Gait, ddof=1)
        sdsd = np.std(diff, ddof=1)
        sd1 = np.sqrt(0.5 * (sdsd ** 2))
        sd2 = np.sqrt(2 * (sd ** 2) - 0.5 * (sdsd ** 2))
        r = sd1 / sd2

        Gait_SD1.append(sd1)
        Gait_SD2.append(sd2)
        Gait_Ratio.append(r)

        # GaitX
        Max_Value_X.append(GaitX.max())
        Gait_Min_X.append(GaitX.min())
        Diff_max_min_X.append(GaitX.max() - GaitX.min())
        Gait_STD_X.append(statistics.stdev(GaitX))
        Gait_Mean_X.append(GaitX.mean())
        Gait_Variance_X.append(GaitX.var())
        Gait_Norm_X.append(LA.norm(GaitX))
        Gait_Norm1_X.append(LA.norm(GaitX, 1))
        MaxFFT_X.append(max(abs(fft(GaitX))))
        Gait_Skewness_X.append(skew(GaitX))
        Gait_Kurtosis_X.append(kurtosis(GaitX))
        RMS_X.append(np.sqrt(mean(square(GaitX))))  # rms
        Energy_X.append(np.sum(square(GaitX)) * 1000)
        ZCR_X.append(librosa.feature.zero_crossing_rate(GaitX)[0, 0])

        # GaitY
        Max_Value_Y.append(GaitY.max())
        Gait_Min_Y.append(GaitY.min())
        Diff_max_min_Y.append(GaitY.max() - GaitY.min())
        Gait_STD_Y.append(statistics.stdev(GaitY))
        Gait_Mean_Y.append(GaitY.mean())
        Gait_Variance_Y.append(GaitY.var())
        Gait_Norm_Y.append(LA.norm(GaitY))
        Gait_Norm1_Y.append(LA.norm(GaitY, 1))
        MaxFFT_Y.append(max(abs(fft(GaitY))))
        Gait_Skewness_Y.append(skew(GaitY))
        Gait_Kurtosis_Y.append(kurtosis(GaitY))
        RMS_Y.append(np.sqrt(mean(square(GaitY))))  # rms
        Energy_Y.append(np.sum(square(GaitY)) * 1000)
        ZCR_Y.append(librosa.feature.zero_crossing_rate(GaitY)[0, 0])

        # GaitZ
        Max_Value_Z.append(GaitZ.max())
        Gait_Min_Z.append(GaitZ.min())
        Diff_max_min_Z.append(GaitZ.max() - GaitZ.min())
        Gait_STD_Z.append(statistics.stdev(GaitZ))
        Gait_Mean_Z.append(GaitZ.mean())
        Gait_Variance_Z.append(GaitZ.var())
        Gait_Norm_Z.append(LA.norm(GaitZ))
        Gait_Norm1_Z.append(LA.norm(GaitZ, 1))
        MaxFFT_Z.append(max(abs(fft(GaitZ))))
        Gait_Skewness_Z.append(skew(GaitZ))
        Gait_Kurtosis_Z.append(kurtosis(GaitZ))
        RMS_Z.append(np.sqrt(mean(square(GaitZ))))  # rms
        Energy_Z.append(np.sum(square(GaitZ)) * 1000)
        ZCR_Z.append(librosa.feature.zero_crossing_rate(GaitZ)[0, 0])

        Corr_X_Y.append(LA.norm(np.corrcoef(GaitX, GaitY)))
        Corr_X_Z.append(LA.norm(np.corrcoef(GaitX, GaitZ)))
        Corr_Y_Z.append(LA.norm(np.corrcoef(GaitY, GaitZ)))

        # if tm < (len(a) / fs + start):
        #     tm += step / fs

    # hist = np.histogram(FFT_all, bins=15)
    # fmax: maximal frequency to plot in the histogram in Hz

    ACM_feats = {'timestamp': timestamp,
                 'Max': Max_Value, 'Min': Gait_Min, 'STD': Gait_STD, 'Mean': Gait_Mean, 'Variance': Gait_Variance,
                 'Norm': Gait_Norm, 'Norm1': Gait_Norm1, 'Skewness': Gait_Skewness, 'Kurtosis': Gait_Kurtosis,
                 'Diff_max_min': Diff_max_min, 'MaxFFT': MaxFFT, 'RMS': RMS, 'Energy': Energy, 'ZCR': ZCR,
                 'SD1': Gait_SD1, 'SD2': Gait_SD2, 'Ratio': Gait_Ratio,

                 'Max_X': Max_Value_X, 'Min_X': Gait_Min_X, 'STD_X': Gait_STD_X, 'Mean_X': Gait_Mean_X,
                 'Diff_max_min_X': Diff_max_min_X, 'Variance_X': Gait_Variance_X, 'Norm_X': Gait_Norm_X,
                 'Norm1_X': Gait_Norm1_X, 'Skewness_X': Gait_Skewness_X, 'Kurtosis_X': Gait_Kurtosis_X,
                 'MaxFFT_X': MaxFFT_X, 'RMS_X': RMS_X, 'Energy_X': Energy_X, 'ZCR_X': ZCR_X,

                 'Max_Y': Max_Value_Y, 'Min_Y': Gait_Min_Y, 'STD_Y': Gait_STD_Y, 'Mean_Y': Gait_Mean_Y,
                 'Diff_max_min_Y': Diff_max_min_Y, 'Variance_Y': Gait_Variance_Y, 'Norm_Y': Gait_Norm_Y,
                 'Norm1_Y': Gait_Norm1_Y, 'Skewness_Y': Gait_Skewness_Y, 'Kurtosis_Y': Gait_Kurtosis_Y,
                 'MaxFFT_Y': MaxFFT_Y, 'RMS_Y': RMS_Y, 'Energy_Y': Energy_Y, 'ZCR_Y': ZCR_Y,

                 'Max_Z': Max_Value_Z, 'Min_Z': Gait_Min_Z, 'STD_Z': Gait_STD_Z, 'Mean_Z': Gait_Mean_Z,
                 'Diff GaitZ max-min': Diff_max_min_Z, 'Variance_Z': Gait_Variance_Z, 'Norm_Z': Gait_Norm_Z,
                 'Norm1_Z': Gait_Norm1_Z, 'Skewness_Z': Gait_Skewness_Z, 'Kurtosis_Z': Gait_Kurtosis_Z,
                 'MaxFFT_Z': MaxFFT_Z, 'RMS_Z': RMS_Z, 'Energy_Z': Energy_Z, 'ZCR_Z': ZCR_Z,

                 'Correlation_(X,Y)': Corr_X_Y, 'Correlation_(X,Z)': Corr_X_Z, 'Correlation_(Y,Z)': Corr_Y_Z}

    ACM_feats = pd.DataFrame.from_dict(ACM_feats, orient="index").T.add_prefix("Gait_")

    if plot == 1:
        """plot absolute gait features"""
        # fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 8))
        fig, ax_left = plt.subplots(figsize=(20, 6))
        ax_right = ax_left.twinx()

        ax_right.plot(timestamp, Max_Value, label='Max_Value')
        ax_right.plot(timestamp, Gait_Min, label='Gait_Min')
        ax_right.plot(timestamp, Diff_max_min, label='Diff_max_min')
        ax_right.plot(timestamp, Gait_STD, label='Gait_STD')
        ax_right.plot(timestamp, Gait_Mean, label='Gait_Mean')
        ax_right.plot(timestamp, Gait_Variance, label='Gait_Variance')
        ax_right.plot(timestamp, Gait_Norm, label='Gait_Norm')
        ax_right.plot(timestamp, Gait_Norm1, label='Gait_Norm1')
        ax_right.plot(timestamp, MaxFFT, label='MaxFFT')
        ax_right.plot(timestamp, Gait_Skewness, label='Gait_Skewness')
        ax_right.plot(timestamp, Gait_Kurtosis, label='Gait_Kurtosis')
        ax_right.plot(timestamp, RMS, label='RMS')
        ax_right.plot(timestamp, Energy, label='Energy')
        ax_right.plot(timestamp, ZCR, label='ZCR')
        ax_right.plot(timestamp, Gait_SD1, label='Gait_SD1')
        ax_left.plot(timestamp, Gait_SD2, label='Gait_SD2')
        ax_right.plot(timestamp, Gait_Ratio, label='Gait_Ratio')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.title("Acceleration features on norm A")
        # ax2.hist(hist[1][:-1], hist[1], weights=hist[0], rwidth  = 0.5);
        # ax2.set(xlabel='frequency (Hz)', ylabel='count')
        # ax2.set_title("Histogram of acceleration frequencies")
        plt.show()

    elif plot == 2:
        """plot gait features on X axis"""
        # fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 8))

        fig, ax_left = plt.subplots(figsize=(20, 6))
        ax_right = ax_left.twinx()

        ax_right.plot(timestamp, Max_Value_X, label='Max_Value')
        ax_right.plot(timestamp, Gait_Min_X, label='Gait_Min')
        ax_right.plot(timestamp, Diff_max_min_X, label='Diff_max_min')
        ax_right.plot(timestamp, Gait_STD_X, label='Gait_STD')
        ax_right.plot(timestamp, Gait_Mean_X, label='Gait_Mean')
        ax_right.plot(timestamp, Gait_Variance_X, label='Gait_Variance')
        ax_right.plot(timestamp, Gait_Norm_X, label='Gait_Norm')
        ax_right.plot(timestamp, Gait_Norm1_X, label='Gait_Norm1')
        ax_right.plot(timestamp, MaxFFT_X, label='MaxFFT')
        ax_right.plot(timestamp, Gait_Skewness_X, label='Gait_Skewness')
        ax_right.plot(timestamp, Gait_Kurtosis_X, label='Gait_Kurtosis')
        ax_right.plot(timestamp, RMS_X, label='RMS')
        ax_left.plot(timestamp, Energy_X, label='Energy')
        ax_right.plot(timestamp, ZCR_X, label='ZCR')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.title("Acceleration features on Ax")
        # ax2.hist(hist[1][:-1], hist[1], weights=hist[0], rwidth  = 0.5);
        # ax2.set(xlabel='frequency (Hz)', ylabel='count')
        # ax2.set_title("Histogram of acceleration frequencies")
        plt.show()

    elif plot == 3:
        """plot gait features on Y axis"""
        # fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 8))
        fig, ax_left = plt.subplots(figsize=(20, 6))
        ax_right = ax_left.twinx()

        ax_right.plot(timestamp, Max_Value_Y, label='Max_Value')
        ax_right.plot(timestamp, Gait_Min_Y, label='Gait_Min')
        ax_right.plot(timestamp, Diff_max_min_Y, label='Diff_max_min')
        ax_right.plot(timestamp, Gait_STD_Y, label='Gait_STD')
        ax_right.plot(timestamp, Gait_Mean_Y, label='Gait_Mean')
        ax_right.plot(timestamp, Gait_Variance_Y, label='Gait_Variance')
        ax_right.plot(timestamp, Gait_Norm_Y, label='Gait_Norm')
        ax_right.plot(timestamp, Gait_Norm1_Y, label='Gait_Norm1')
        ax_right.plot(timestamp, MaxFFT_Y, label='MaxFFT')
        ax_right.plot(timestamp, Gait_Skewness_Y, label='Gait_Skewness')
        ax_right.plot(timestamp, Gait_Kurtosis_Y, label='Gait_Kurtosis')
        ax_right.plot(timestamp, RMS_Y, label='RMS')
        ax_left.plot(timestamp, Energy_Y, label='Energy')
        ax_right.plot(timestamp, ZCR_Y, label='ZCR')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.title("Acceleration features on Ay")
        # ax2.hist(hist[1][:-1], hist[1], weights=hist[0], rwidth  = 0.5);
        # ax2.set(xlabel='frequency (Hz)', ylabel='count')
        # ax2.set_title("Histogram of acceleration frequencies")
        plt.show()

    elif plot == 4:
        """plot gait features on Z axis"""
        # fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 8))
        fig, ax_left = plt.subplots(figsize=(20, 6))
        ax_right = ax_left.twinx()

        ax_right.plot(timestamp, Max_Value_Z, label='Max_Value')
        ax_right.plot(timestamp, Gait_Min_Z, label='Gait_Min')
        ax_right.plot(timestamp, Diff_max_min_Z, label='Diff_max_min')
        ax_right.plot(timestamp, Gait_STD_Z, label='Gait_STD')
        ax_right.plot(timestamp, Gait_Mean_Z, label='Gait_Mean')
        ax_right.plot(timestamp, Gait_Variance_Z, label='Gait_Variance')
        ax_right.plot(timestamp, Gait_Norm_Z, label='Gait_Norm')
        ax_right.plot(timestamp, Gait_Norm1_Z, label='Gait_Norm1')
        ax_right.plot(timestamp, MaxFFT_Z, label='MaxFFT')
        ax_right.plot(timestamp, Gait_Skewness_Z, label='Gait_Skewness')
        ax_right.plot(timestamp, Gait_Kurtosis_Z, label='Gait_Kurtosis')
        ax_right.plot(timestamp, RMS_Z, label='RMS')
        ax_left.plot(timestamp, Energy_Z, label='Energy')
        ax_right.plot(timestamp, ZCR_Z, label='ZCR')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.title("Acceleration features on Az")
        # ax2.hist(hist[1][:-1], hist[1], weights=hist[0], rwidth  = 0.5);
        # ax2.set(xlabel='frequency (Hz)', ylabel='count')
        # ax2.set_title("Histogram of acceleration frequencies")
        plt.show()

    else:
        pass

    return ACM_feats


def plot_ACM_features(timestamp, df_acm_features, seizure_onset,  seizure_offset, diff,
                      show=True, save_fig=None):
    """
    timestamp : timescale to use for plot, in seconds, formatted to utc timezone
    df_acm_features: dataframe of ACM features
    seizure_onset: seizure onset in seconds, as annotated in EDF
    diff : edf file start time in seconds
    """

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set(xlabel="Time (UTC)", title="ACM features")
    ax.plot(timestamp, normalize(df_acm_features['Gait_Max']), label='Max')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Min']), label='Min')
    ax.plot(timestamp, normalize(df_acm_features['Gait_STD']), label='STD')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Mean']), label='Mean')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Variance']), label='Variance')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Norm']), label='Norm')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Norm1']), label='Norm1')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Skewness']), label='Skewness')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Kurtosis']), label='Kurtosis')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Diff_max_min']), label='Diff_max_min')
    ax.plot(timestamp, normalize(df_acm_features['Gait_MaxFFT']), label='MaxFFT')
    ax.plot(timestamp, normalize(df_acm_features['Gait_RMS']), label='RMS')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Energy']), label='Energy')
    ax.plot(timestamp, normalize(df_acm_features['Gait_ZCR']), label='ZCR')
    ax.plot(timestamp, normalize(df_acm_features['Gait_SD1']), label='SD1')
    ax.plot(timestamp, normalize(df_acm_features['Gait_SD2']), label='SD2')
    ax.plot(timestamp, normalize(df_acm_features['Gait_Ratio']), label='Ratio')
    ax.axvline(seizure_onset + diff.seconds, label='seizure onset', c='r', alpha=.6)
    ax.axvline(seizure_offset + diff.seconds, label='seizure end', c='r', alpha=.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
    ax.xaxis.set_major_formatter(formatter)
    
    if save_fig:
        plt.savefig(save_fig+'_acm.png', dpi=400, bbox_inches='tight')
        
    if show:
        plt.show()


def plot_ACM_features_record(timestamp, df_acm_features,
                             df_records, record_row,
                             show=True, save_fig=None):
    """
    timestamp : timescale to use for plot, in seconds, formatted to utc timezone
    df_acm_features: dataframe of ACM features
    df_records: dataframe of records details
    record_row : index of the selected record in the df_records dataframe
    """
    rec_sd = datetime.datetime.strptime(df_records.Record_start_date[record_row], '%Y-%m-%d %H:%M:%S%z')
    dt_x = datetime.datetime.combine(rec_sd.date(), datetime.datetime.min.time())
    dt_x = dt_x.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
    diff = rec_sd - dt_x

    # r_start = 0
    # r_end = dt_x.fromisoformat(df_records.Record_end_date[record_row]) - dt_x.fromisoformat(
    #     df_records.Record_start_date[record_row])
    # hh, mm, ss = 0, 0, r_end.seconds
    # r_end = (hh * 3600 + mm * 60 + ss)

    Desc = df_records.Descriptions[record_row]
    Desc = Desc.replace("[", "").replace("]", "").replace("'", "").split(', ') if not (pd.isna(Desc)) else []
    Onsets = df_records.Onsets[record_row]
    Onsets = Onsets.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Onsets)) else []
    Onsets = [int(x) for x in Onsets]
    Durations = df_records.Durations[record_row]
    Durations = Durations.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Durations)) else []
    Durations = [int(x) for x in Durations]

    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))

    fig, ax = plt.subplots(figsize=(19, 4))
    ax.set(xlabel="Time (UTC)", title="ACM features")
    for i in df_acm_features.columns[1:18]:
        ax.plot(timestamp, normalize(df_acm_features[i]), label=i[5:])

    for i in range(len(Desc)):
        sz_onset = Onsets[i] + diff.seconds
        sz_end = Onsets[i] + Durations[i] + diff.seconds
        if Desc[i] == 'FBTCS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='red', label='FBTCS')
        elif Desc[i] == 'FIAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='skyblue', label='FIAS')
        elif Desc[i] == 'FAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='grey', label='FAS')
        elif Desc[i] == 'FUAS':
            ax.axvspan(sz_onset, sz_end, alpha=0.5, color='powderblue', label='FUAS')
    ax.xaxis.set_major_formatter(formatter)
    # modules.legend_without_duplicate_labels(ax)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    if show:
        plt.show()
    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')

# -*- coding: utf-8 -*-
"""
Module for HRV features extracted from R-peaks of the ECG in temporal, spectral and non-linear domains 

@author: Oumayma Gharbi
"""

from utils import *
from utils import modules
from .slidingWindow import slidingWindow

# from matplotlib import style
# style.use('ggplot')

# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace
# pylint: disable=pointless-string-statement
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long


def normalize(v):
    """Normalization function"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def timeDomain_features(Rpeaks, win_size, step, plot=0, Normalize=True):
    """ Computes time domain features on RR interval data
    The time domain features calculated are chosen according to:

            Shaffer and Ginsberg, An Overview of Heart Rate Variability Metrics and Norms, 2017
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/pdf/fpubh-05-00258.pdf

    This fonctions uses Slidingwindow module to identify extraction windows based on the number of RRIs detected
    Parameters:
    ------------
    rri : list, shape = [n_samples,]
           RR interval data

    win_size: Number of sample points of the RRint series in the feature extraction window.

    step: Number of points to overlap between segments.

    plot : int, 0|1
          Setting plot to 1 creates a matplotlib figure showing time domain features
          extracted from RR interval series.

    Returns:
    ---------
    timeDomainFeats : dict
                   MeanRR, RMSSD SDNN, SDSD, VAR, NN50, pNN50, NN20, pNN20, HTI, TINN
    """
    rri = np.diff(Rpeaks)

    timestamp = []
    MeanRR = []
    RMSSD = []
    SDNN = []
    SDSD = []
    VAR = []
    NN50 = []
    NN20 = []
    pNN50 = []
    pNN20 = []
    HTI = []
    TINN = []

    chunklist = list(slidingWindow(rri, win_size, step))

    numOfChunks = ((len(rri)-win_size)/step)+1
    i = win_size
    tm = Rpeaks[i]

    for sublist in chunklist:
        diff_nni = np.diff(sublist)
        N = len(sublist)

        # Mean-based
        meanrr = np.mean(sublist)
        rmssd = np.sqrt(np.mean(diff_nni ** 2))
        sdnn = np.std(sublist, ddof=1)
        sdsd = np.std(diff_nni, ddof=1)
        var = statistics.variance(sublist)

        # Extreme-based
        nn50 = np.sum(np.abs(diff_nni) > 50)
        nn20 = np.sum(np.abs(diff_nni) > 20)
        pnn50 = 100 * nn50 / N
        pnn20 = 100 * nn20 / N

        # Geometric domain
        bar_y, bar_x = np.histogram(sublist, bins="auto")
        # HRV Triangular Index
        hti = N / np.max(bar_y)  
        # Triangular Interpolation of the NN Interval Histogram
        tinn = np.max(bar_x) - np.min(bar_x)  

        timestamp.append(tm)
        MeanRR.append(meanrr)
        RMSSD.append(rmssd)
        SDNN.append(sdnn)
        SDSD.append(sdsd)
        VAR.append(var)
        NN50.append(nn50)
        NN20.append(nn20)
        pNN50.append(pnn50)
        pNN20.append(pnn20)
        HTI.append(hti)
        TINN.append(tinn)

        i += step
        if (i < win_size+int(numOfChunks)*step) and (i < len(Rpeaks)):
            tm = Rpeaks[i]

    if Normalize:
        MeanRR = normalize(MeanRR)
        RMSSD = normalize(RMSSD)
        SDNN = normalize(SDNN)
        SDSD = normalize(SDSD)
        VAR = normalize(VAR)
        NN50 = normalize(NN50)
        NN20 = normalize(NN20)
        pNN50 = normalize(pNN50)
        pNN20 = normalize(pNN20)
        HTI = normalize(HTI)
        TINN = normalize(TINN)

    timeDomainFeats = {'timestamp': timestamp, 'MeanRR': MeanRR, 'RMSSD': RMSSD, 'SDNN': SDNN,
                       'SDSD': SDSD, 'VAR': VAR, 'NN50': NN50, 'pNN50': pNN50, 'NN20': NN20,
                       'pNN20': pNN20, 'HTI': HTI, 'TINN': TINN}

    timeDomainFeats = pd.DataFrame.from_dict(timeDomainFeats, orient="index").T.add_prefix("HRV_")

    if plot == 1:
        plt.figure(figsize=(20, 4))
        plt.title('Time features')
        plt.plot(timestamp, MeanRR, label='MeanRR')
        plt.plot(timestamp, RMSSD, label='RMSSD')
        plt.plot(timestamp, SDNN, label='SDNN')
        plt.plot(timestamp, SDSD, label='SDSD')
        plt.plot(timestamp, VAR, label='VAR')
        plt.plot(timestamp, NN50, label='NN50')
        plt.plot(timestamp, pNN50, label='pNN50')
        plt.plot(timestamp, NN20, label='NN20')
        plt.plot(timestamp, pNN20, label='pNN20')
        plt.plot(timestamp, HTI, label='HTI')
        plt.plot(timestamp, TINN, label='TINN')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.show()
        
    else:
        pass
    
    return timeDomainFeats


def r_peaks_in_window(win_start, win_end, r_peaks):
    """"Extract R-Peaks positions found in a specified time window"""
    out = [x for x in r_peaks if (x >= win_start) and (x <= win_end)]
    return np.array(out)


def timeDomain_features_time(Rpeaks, start, end, win_size, step, fs, plot=0, Normalize=True):
    """ Computes time domain features on RR interval data
    The time domain features calculated are chosen according to:

            Shaffer and Ginsberg, An Overview of Heart Rate Variability Metrics and Norms, 2017
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/pdf/fpubh-05-00258.pdf

    This fonction calculates features based on a temporal window, not on the number of RRIs
    if noisy or bad ECG channel, where R-peaks are not detected, it will simply not calculate features
    Parameters:
    ------------
    rri : list, shape = [n_samples,]
           RR interval data
    win_size: Window size in seconds
    step: in seconds
    fs: sampling frequency
    plot : int, 0|1
          Setting plot to 1 creates a matplotlib figure showing time domain features
          extracted from RR interval series.

    Returns:
    ---------
    timeDomainFeats : dict
                   MeanRR, RMSSD SDNN, SDSD, VAR, NN50, pNN50, NN20, pNN20, HTI, TINN
    """

    timestamp = []
    MeanRR = []
    RMSSD = []
    SDNN = []
    SDSD = []
    VAR = []
    NN50 = []
    NN20 = []
    pNN50 = []
    pNN20 = []
    HTI = []
    TINN = []
    
    # rri = np.diff(Rpeaks)
    # chunklist = list(slidingWindow(rri,win_size,step))
    # numOfChunks = ((len(rri)-win_size)/step)+1
    # i = win_size
    # tm = Rpeaks[i]
    
    # NOTE: temporal interval is considred, independent of number of RRIs
    # This is a more realistic implementation in case of noisy or bad ECG channel
    # where QRS morphology is undetectable

    # for i in range(int(Rpeaks[0]), int(Rpeaks[-1]), step * fs):
    for i in range(start*fs, end*fs, step * fs):
        # d.append([i, i + win_size * fs])
        winStart = i
        winEnd = i + win_size * fs
        Rpeaks_crop = r_peaks_in_window(winStart, winEnd, Rpeaks)
        sublist = np.diff(Rpeaks_crop)
    
    # for sublist in chunklist:
        diff_nni = np.diff(sublist)
        N = len(sublist)
        
        if N > 2:
            # Mean-based
            meanrr = np.mean(sublist)
            rmssd = np.sqrt(np.mean(diff_nni ** 2))
            sdnn = np.std(sublist, ddof=1)
            sdsd = np.std(diff_nni, ddof=1)
            var = statistics.variance(sublist)
    
            # Extreme-based
            nn50 = np.sum(np.abs(diff_nni) > 50)
            nn20 = np.sum(np.abs(diff_nni) > 20)
            pnn50 = 100 * nn50 / N
            pnn20 = 100 * nn20 / N
    
            # Geometric domain
            bar_y, bar_x = np.histogram(sublist, bins="auto")
            # #HRV Triangular Index
            hti = N / np.max(bar_y)  
            # #Triangular Interpolation of the NN Interval Histogram
            tinn = np.max(bar_x) - np.min(bar_x)  
    
            timestamp.append(i + win_size * fs)
            MeanRR.append(meanrr)
            RMSSD.append(rmssd)
            SDNN.append(sdnn)
            SDSD.append(sdsd)
            VAR.append(var)
            NN50.append(nn50)
            NN20.append(nn20)
            pNN50.append(pnn50)
            pNN20.append(pnn20)
            HTI.append(hti)
            TINN.append(tinn)

        # i += step
        # if (i < win_size+int(numOfChunks)*step) and (i<len(Rpeaks)):
            # tm = Rpeaks[i]

    # Normalize only for plotting !!!!!!!
    # for training and testing, should not normalize since it takes info from future
    if Normalize and plot == 1:
        MeanRR = normalize(MeanRR)
        RMSSD = normalize(RMSSD)
        SDNN = normalize(SDNN)
        SDSD = normalize(SDSD)
        VAR = normalize(VAR)
        NN50 = normalize(NN50)
        NN20 = normalize(NN20)
        pNN50 = normalize(pNN50)
        pNN20 = normalize(pNN20)
        HTI = normalize(HTI)
        TINN = normalize(TINN)

    timeDomainFeats = {'timestamp': timestamp, 'MeanRR': MeanRR, 'RMSSD': RMSSD, 'SDNN': SDNN,
                       'SDSD': SDSD, 'VAR': VAR, 'NN50': NN50, 'pNN50': pNN50, 'NN20': NN20,
                       'pNN20': pNN20, 'HTI': HTI, 'TINN': TINN}

    timeDomainFeats = pd.DataFrame.from_dict(timeDomainFeats, orient="index").T.add_prefix("HRV_")

    if plot == 1:
        plt.figure(figsize=(20, 4))
        plt.title('Time features')
        plt.plot(timestamp, MeanRR, label='MeanRR')
        plt.plot(timestamp, RMSSD, label='RMSSD')
        plt.plot(timestamp, SDNN, label='SDNN')
        plt.plot(timestamp, SDSD, label='SDSD')
        plt.plot(timestamp, VAR, label='VAR')
        plt.plot(timestamp, NN50, label='NN50')
        plt.plot(timestamp, pNN50, label='pNN50')
        plt.plot(timestamp, NN20, label='NN20')
        plt.plot(timestamp, pNN20, label='pNN20')
        plt.plot(timestamp, HTI, label='HTI')
        plt.plot(timestamp, TINN, label='TINN')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.show()
        
    else:
        pass
    
    return timeDomainFeats


def freqDomain_features(Rpeaks, win_size, overlap, lf_bw=0.11, hf_bw=0.1, plot=0):
    """ Computes frequency domain features on RR interval data

    Parameters:
    ------------
    RRints : list, shape = [n_samples,]
           RR interval data

    lf_bw : float, optional
          Low frequency bandwidth centered around LF band peak frequency
          when band_type is set to 'adapted'. Defaults to 0.11
          
    hf_bw : float, optional
          High frequency bandwidth centered around HF band peak frequency
          when band_type is set to 'adapted'. Defaults to 0.1
    plot : int, 1|0
          Setting plot to 1 creates a matplotlib figure showing frequency
          versus spectral power with color shading to indicate the VLF, LF,
          and HF band bounds.
          
    win_size: Number of sample points of the RRint seriesin the feature extraction window.
    
    overlap: Number of points to overlap between segments. 
    
    Returns:
    ---------
    freqDomainFeats : dict
                   VLF_Power, LF_Power, HF_Power, LF/HF Ratio
    """

    RRints = np.diff(Rpeaks)

    # Remove ectopic beats
    # RR intervals differing by more than 20% from the one proceeding it are removed
    NNs = []
    for c, rr in enumerate(RRints):
        if abs(rr - RRints[c-1]) <= 0.20 * RRints[c-1]:
            NNs.append(rr)

    # Resample @ 4 Hz
    fsResamp = 4   
    vlf_NU_log = 0
    lf_NU_log = 0
    hf_NU_log = 0
    lfhfRation_log = 0
    
    vlf = (0.003, 0.04)
    lf = (0.04, 0.15)
    hf = (0.15, 0.4)
    
    if len(NNs) > 0:
        tmStamps = np.cumsum(NNs)  # in seconds
        f = interpolate.interp1d(tmStamps, NNs, 'cubic')
        tmInterp = np.arange(tmStamps[0], tmStamps[-1], 1/fsResamp)
        RRinterp = f(tmInterp)
        
    # Remove DC component
        RRseries = RRinterp - np.mean(RRinterp)
    
    # Pwelch w/ zero pad
        fxx, pxx = signal.welch(RRseries, fsResamp, window='hanning',
                                nperseg=win_size, noverlap=overlap, nfft=win_size)

        plot_labels = ['VLF', 'LF', 'HF']

        df = fxx[1] - fxx[0]
        vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])], dx=df)
        lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])], dx=df)
        hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])], dx=df)
        totalPower = vlf_power + lf_power + hf_power

    # Normalize and take log
        vlf_NU_log = np.log((vlf_power / (totalPower - vlf_power)) + 1)
        lf_NU_log = np.log((lf_power / (totalPower - vlf_power)) + 1)
        hf_NU_log = np.log((hf_power / (totalPower - vlf_power)) + 1)
        lfhfRation_log = np.log((lf_power / hf_power) + 1)
        
        if plot == 1:
            # Plot option
            freq_bands = {'vlf': vlf, 'lf': lf, 'hf': hf}
            freq_bands = OrderedDict(sorted(freq_bands.items(), key=lambda t: t[0]))
            colors = ['lightsalmon', 'lightsteelblue', 'darkseagreen']
            _, ax = plt.subplots(1)
            ax.plot(fxx, pxx, c='grey')
            plt.title('HRV Frequency features')
            plt.xlim([0, 0.40])
            plt.xlabel(r'Frequency $(Hz)$')
            plt.ylabel(r'PSD $(s^2/Hz$)')

            for c, key in enumerate(freq_bands):
                a = fxx >= freq_bands[key][0]
                b = fxx <= freq_bands[key][1]
                ax.fill_between(fxx[min(np.where(a)[0]): max(np.where(b)[0])],
                                pxx[min(np.where(a)[0]): max(np.where(b)[0])],
                                0, facecolor=colors[c])
                
            patch1 = mpatches.Patch(color=colors[0], label=plot_labels[2])
            patch2 = mpatches.Patch(color=colors[1], label=plot_labels[1])
            patch3 = mpatches.Patch(color=colors[2], label=plot_labels[0])
            plt.legend(handles=[patch1, patch2, patch3])
            plt.show()
            
        else:
            pass
        
    freqDomainFeats = {'timestamp': Rpeaks[0], 'VLF_Power': vlf_NU_log, 'LF_Power': lf_NU_log,
                       'HF_Power': hf_NU_log, 'LF/HF': lfhfRation_log}

    return freqDomainFeats


def nonLinearDomain_features(Rpeaks, win_size, step, plot=0, Normalize=True):
    """Computes time domain features on RR interval data
    the non-linear features are computed according to :

        Karmakar et al. 'Complex Correlation Measure: a novel descriptor for Poincaré plot',
        August 2009.
        https://rdcu.be/b3FNK

    This fonction calculates features based on a window of RRIs
    Parameters:
    ------------
    rri : list, shape = [n_samples,]
           RR interval data

    win_size: Number of sample points of the RRint series in the feature extraction window.

    step: Number of points to overlap between segments.

    plot : int, 0|1
          Setting plot to 1 creates a matplotlib figure showing non-linear domain features
          extracted from RR interval series.

    Returns:
    ---------
    nonLinearFeats : dict
                   SD1, SD2, CSI, CVI, ratio_SD1_SD2, SampEn, CoSEn, KFD
    """
    rri = np.diff(Rpeaks)

    timestamp = []
    SD1 = []
    SD2 = []
    ratio_SD1_SD2 = []
    CSI = []
    CVI = []
    SampEn = []
    CoSEn = []
    KFD = []

    chunklist = list(slidingWindow(rri, win_size, step))

    numOfChunks = ((len(rri) - win_size) / step) + 1
    i = win_size
    tm = Rpeaks[i]

    for sublist in chunklist:
        diff_nni = np.diff(sublist)

        """
        Karmakar et al, 'Complex Correlation Measure: a novel descriptor for Poincaré plot'
        """
        sdnn = np.std(sublist, ddof=1)
        sdsd = np.std(diff_nni, ddof=1)

        try:
            sd1 = np.sqrt(0.5 * (sdsd ** 2))
            sd2 = np.sqrt(2 * (sdnn ** 2) - 0.5 * (sdsd ** 2))
        except ValueError:
            print('Negative value passed to sqrt()')
            # sd1, sd2 = 0, 0 ##############################################

        """An Overview of Heart Rate Variability Metrics and Norms
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/pdf/fpubh-05-00258.pdf
        SD1/SD2 measures the unpredictability of the RR time series"""
        try:
            ratio_sd1_sd2 = sd1 / sd2
        except ZeroDivisionError:
            print("SD2 is 0, can't calculate ratio")
            # ratio_sd1_sd2 = 0 ##############################################

        """https://doi.org/10.1371/journal.pone.0204339"""
        try:
            csi = sd2 / sd1
        except ZeroDivisionError:
            print("SD1 is 0, can't calculate ratio")
            # csi = 0 ##############################################

        try:
            assert sd1*sd2 > 0
            cvi = np.log10(sd1 * sd2)
        except AssertionError:
            print("SD1*SD2 is negative, can't calculate CVI")
            # cvi = 0 ##############################################

        """Lake and Moorman, Accurate estimation of entropy in very short physiological time series,
        2011"""
        # std_ts = np.std(sublist)
        # sampen_of_series = ent.sample_entropy(sublist, 4, 0.2 * std_ts)
        tolerance = 0.2  # param r: Tolerance. Typically 0.1 or 0.2.
        sampen_of_series = sampen2(sublist, mm=0, r=tolerance)
        sampen = sampen_of_series[0][1]
        if sampen is None:
            sampen = 0
        meanrr = np.mean(sublist)
        cosen = sampen + np.log(2 * tolerance) - np.log(meanrr)

        """KFD: Katz Fractal Dimension  https://www.ncbi.nlm.nih.gov/pubmed/3396335"""
        N = len(sublist)
        if N > 2:
            L = sum([np.sqrt(1 + ((sublist[i - 1] - sublist[i]) ** 2)) for i in range(1, N)])
            d = max([np.sqrt(((0 - i) ** 2) + ((sublist[0] - sublist[i]) ** 2)) for i in range(1, N)])
            kfd = np.log(N - 1) / (np.log(N - 1) + np.log(d / L))

        timestamp.append(tm)
        SD1.append(sd1)
        SD2.append(sd2)
        CSI.append(csi)
        CVI.append(cvi)
        ratio_SD1_SD2.append(ratio_sd1_sd2)
        # SampEn.append(sampen_of_series[0][1])
        SampEn.append(sampen)
        CoSEn.append(cosen)
        KFD.append(kfd)

        i += step
        if (i < win_size + int(numOfChunks) * step) and (i < len(Rpeaks)):
            tm = Rpeaks[i]

    if Normalize:
        SD1 = normalize(SD1)
        SD2 = normalize(SD2)
        CSI = normalize(CSI)
        CVI = normalize(CVI)
        KFD = normalize(KFD)
        CoSEn = normalize(CoSEn)
        SampEn = normalize(SampEn)
        ratio_SD1_SD2 = normalize(ratio_SD1_SD2)
        SampEn = normalize(SampEn)

    nonLinearFeats = {'timestamp': timestamp, 'SD1': SD1, 'SD2': SD2, 'CSI': CSI,
                      'CVI': CVI, 'ratio_SD1_SD2': ratio_SD1_SD2,
                      'SampEn': SampEn, 'CoSEn': CoSEn, 'KFD': KFD}

    if plot == 1:
        plt.figure(figsize=(20, 4))
        plt.title('Non-linear domain features')
        plt.plot(timestamp, SD1, label='SD1')
        plt.plot(timestamp, SD2, label='SD2')
        plt.plot(timestamp, CSI, label='CSI')
        plt.plot(timestamp, CVI, label='CVI')
        plt.plot(timestamp, ratio_SD1_SD2, label='ratio_SD1_SD2')
        plt.plot(timestamp, SampEn, label='SampEn mod')
        plt.plot(timestamp, CoSEn, label='CoSEn')
        plt.plot(timestamp, KFD, label='KFD')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.show()

    else:
        pass

    nonLinearFeats = pd.DataFrame.from_dict(nonLinearFeats, orient="index").T.add_prefix("HRV_")
    return nonLinearFeats


def nonLinearDomain_features_time(Rpeaks, start, end, win_size, step, fs, plot=0, Normalize=True):
    """Computes time domain features on RR interval data
    the non-linear features are computed according to :

        Karmakar et al. 'Complex Correlation Measure: a novel descriptor for Poincaré plot',
        August 2009.
        https://rdcu.be/b3FNK

    This fonction calculates features based on a temporal window, not on the number of RRIs
    Parameters:
    ------------
    rri : list, shape = [n_samples,]
           RR interval data

    win_size: Window size in seconds

    step: in seconds

    plot : int, 0|1
          Setting plot to 1 creates a matplotlib figure showing non-linear domain features
          extracted from RR interval series.

    Returns:
    ---------
    nonLinearFeats : dict
                   SD1, SD2, CSI, CVI, ratio_SD1_SD2, SampEn, CoSEn, KFD
    """
    # rri = np.diff(Rpeaks)

    timestamp = []
    SD1 = []
    SD2 = []
    ratio_SD1_SD2 = []
    CSI = []
    CVI = []
    SampEn = []
    CoSEn = []
    KFD = []

    # chunklist = list(slidingWindow(rri, win_size, step))

    # numOfChunks = ((len(rri) - win_size) / step) + 1
    # i = win_size
    # tm = Rpeaks[i]

    # for i in range(int(Rpeaks[0]), int(Rpeaks[-1]), step * fs):
    for i in range(start * fs, end * fs, step * fs):
        # d.append([i, i + win_size * fs])
        winStart = i
        winEnd = i + win_size * fs
        Rpeaks_crop = r_peaks_in_window(winStart, winEnd, Rpeaks)
        sublist = np.diff(Rpeaks_crop)

    # for sublist in chunklist:
        diff_nni = np.diff(sublist)
        N = len(sublist)

        if N > 2:
            """
            Karmakar et al, 'Complex Correlation Measure: a novel descriptor for Poincaré plot'
            """
            sdnn = np.std(sublist, ddof=1)
            sdsd = np.std(diff_nni, ddof=1)

            try:
                sd1 = np.sqrt(0.5 * (sdsd ** 2))
                sd2 = np.sqrt(2 * (sdnn ** 2) - 0.5 * (sdsd ** 2))
            except ValueError:
                print('Negative value passed to sqrt()')
                sd1, sd2 = 0, 0  # #############################################

            """An Overview of Heart Rate Variability Metrics and Norms
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/pdf/fpubh-05-00258.pdf
            SD1/SD2 measures the unpredictability of the RR time series"""
            try:
                ratio_sd1_sd2 = sd1 / sd2
            except ZeroDivisionError:
                print("SD2 is 0, can't calculate ratio")
                ratio_sd1_sd2 = 0  # #############################################

            """https://doi.org/10.1371/journal.pone.0204339"""
            try:
                csi = sd2 / sd1
            except ZeroDivisionError:
                print("SD1 is 0, can't calculate ratio")
                csi = 0  # #############################################

            try:
                assert sd1 * sd2 > 0
                cvi = np.log10(sd1 * sd2)
            except AssertionError:
                print("SD1*SD2 is negative, can't calculate CVI")
                cvi = 0

            """Lake and Moorman, Accurate estimation of entropy in very short physiological time series,
            2011"""
            # std_ts = np.std(sublist)
            # sampen_of_series = ent.sample_entropy(sublist, 4, 0.2 * std_ts)
            tolerance = 0.2  # param r: Tolerance. Typically 0.1 or 0.2.
            sampen_of_series = sampen2(sublist, mm=0, r=tolerance)
            sampen = sampen_of_series[0][1]
            if sampen is None:
                sampen = 0
            meanrr = np.mean(sublist)
            cosen = sampen + np.log(2 * tolerance) - np.log(meanrr)

            """KFD: Katz Fractal Dimension  https://www.ncbi.nlm.nih.gov/pubmed/3396335"""
            L = sum([np.sqrt(1 + ((sublist[i - 1] - sublist[i]) ** 2)) for i in range(1, N)])
            d = max([np.sqrt(((0 - i) ** 2) + ((sublist[0] - sublist[i]) ** 2)) for i in range(1, N)])
            kfd = np.log(N - 1) / (np.log(N - 1) + np.log(d / L))

            timestamp.append(i + win_size * fs)
            SD1.append(sd1)
            SD2.append(sd2)
            CSI.append(csi)
            CVI.append(cvi)
            ratio_SD1_SD2.append(ratio_sd1_sd2)
            # SampEn.append(sampen_of_series[0][1])
            SampEn.append(sampen)
            CoSEn.append(cosen)
            KFD.append(kfd)

            # i += step
            # if (i < win_size + int(numOfChunks) * step) and (i < len(Rpeaks)):
            #   # tm = Rpeaks[i]

    if Normalize:
        SD1 = normalize(SD1)
        SD2 = normalize(SD2)
        CSI = normalize(CSI)
        CVI = normalize(CVI)
        KFD = normalize(KFD)
        CoSEn = normalize(CoSEn)
        SampEn = normalize(SampEn)
        ratio_SD1_SD2 = normalize(ratio_SD1_SD2)
        SampEn = normalize(SampEn)

    nonLinearFeats = {'timestamp': timestamp, 'SD1': SD1, 'SD2': SD2, 'CSI': CSI,
                      'CVI': CVI, 'ratio_SD1_SD2': ratio_SD1_SD2,
                      'SampEn': SampEn, 'CoSEn': CoSEn, 'KFD': KFD}

    if plot == 1:
        plt.figure(figsize=(20, 4))
        plt.title('Non-linear domain features')
        plt.plot(timestamp, SD1, label='SD1')
        plt.plot(timestamp, SD2, label='SD2')
        plt.plot(timestamp, CSI, label='CSI')
        plt.plot(timestamp, CVI, label='CVI')
        plt.plot(timestamp, ratio_SD1_SD2, label='ratio_SD1_SD2')
        plt.plot(timestamp, SampEn, label='SampEn mod')
        plt.plot(timestamp, CoSEn, label='CoSEn')
        plt.plot(timestamp, KFD, label='KFD')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.show()

    else:
        pass

    nonLinearFeats = pd.DataFrame.from_dict(nonLinearFeats, orient="index").T.add_prefix("HRV_")
    return nonLinearFeats


def plot_Rpeaks(raw_ecg, R_peaks, fs, start, end, diff, seizure_onset, seizure_offset, show=True, save_fig=None):
    """Plot detected R-peaks on synchronized timescale, with seizure onset/offset"""
    e_time = np.arange(len(raw_ecg)) / fs
    R_peaks_val = raw_ecg[R_peaks]
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))

    fig, ax = plt.subplots(figsize=(20, 4))

    ax.set(xlabel="Time (UTC)",
           ylabel="ECG",
           title="Detected R peaks mne")
    ax.plot(e_time[start * fs:end * fs] + diff.seconds, raw_ecg[start * fs:end * fs], alpha=.5)
    ax.scatter(R_peaks / fs + diff.seconds, R_peaks_val, c='orange')
    ax.axvline(seizure_onset + diff.seconds, label='seizure onset', c='r', alpha=.6)
    ax.axvline(seizure_offset + diff.seconds, label='seizure end', c='r', alpha=.6)
    ax.xaxis.set_major_formatter(formatter)

    if show:
        plt.show()
    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')


def plot_Rpeaks_record(raw_ecg, R_peaks, fs, df_records, record_row, show=True, save_fig=None):
    """Plot detected R-peaks on synchronized timescale, with seizure onset/offset"""
    # record = df_records.Record_name[record_row]

    rec_sd = datetime.datetime.strptime(df_records.Record_start_date[record_row], '%Y-%m-%d %H:%M:%S%z')
    dt_x = datetime.datetime.combine(rec_sd.date(), datetime.datetime.min.time())
    dt_x = dt_x.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
    diff = rec_sd - dt_x

    r_start = 0
    r_end = dt_x.fromisoformat(df_records.Record_end_date[record_row]) - dt_x.fromisoformat(
        df_records.Record_start_date[record_row])
    # hh, mm, ss = 0, 0, r_end.seconds
    # r_end = (hh * 3600 + mm * 60 + ss)
    r_end = int(r_end.total_seconds())

    Desc = df_records.Descriptions[record_row]
    Desc = Desc.replace("[", "").replace("]", "").replace("'", "").split(', ') if not (pd.isna(Desc)) else []
    Onsets = df_records.Onsets[record_row]
    Onsets = Onsets.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Onsets)) else []
    Onsets = [int(x) for x in Onsets]
    Durations = df_records.Durations[record_row]
    Durations = Durations.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Durations)) else []
    Durations = [int(x) for x in Durations]

    e_time = np.arange(len(raw_ecg)) / fs
    R_peaks_val = raw_ecg[R_peaks]
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))

    fig, ax = plt.subplots(figsize=(19, 4))

    ax.set(xlabel="Time (UTC)", ylabel="ECG",
           title="Detected R peaks MNE")
    # ax.plot(e_time[r_start * fs:r_end * fs] + diff.seconds, raw_ecg[r_start * fs:r_end * fs], alpha=.5, label='ECG')
    ax.plot(e_time + diff.seconds, raw_ecg, alpha=.5, label='ECG')
    ax.scatter(R_peaks / fs + diff.seconds, R_peaks_val, c='orange', label='R peaks')

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
    modules.legend_without_duplicate_labels(ax)

    if show:
        plt.show()
    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')


def plot_tHRV_with_timestamp(timestamp, df_tHRV_features, seizure_onset,
                             seizure_offset, diff, show=True, save_fig=None):
    """
    timestamp : timescale to use for plot, in seconds, formatted to utc timezone
    df_tHRV_features: dataframe of temporal HRV features
    seizure_onset: seizure onset in seconds, as annotated in EDF
    diff : edf file start time in seconds
    """
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
    
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set(xlabel="Time (UTC)", title="HRV Time features")
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_MeanRR']), label='MeanRR')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_RMSSD']), label='RMSSD')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_SDNN']), label='SDNN')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_SDSD']), label='SDSD')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_VAR']), label='VAR')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_NN50']), label='NN50')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_pNN50']), label='pNN50')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_NN20']), label='NN20')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_pNN20']), label='pNN20')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_HTI']), label='HTI')
    ax.plot(timestamp, normalize(df_tHRV_features['HRV_TINN']), label='TINN')
    ax.axvline(seizure_onset + diff.seconds, label='seizure onset', c='r', alpha=.6)
    ax.axvline(seizure_offset + diff.seconds, label='seizure end', c='r', alpha=.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    ax.xaxis.set_major_formatter(formatter)
    
    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')
        
    if show:
        plt.show()


def plot_tHRV_with_timestamp_record(timestamp, df_tHRV_features,
                                    df_records, record_row,
                                    show=True, save_fig=None):
    """
    timestamp : timescale to use for plot, in seconds, formatted to utc timezone
    df_tHRV_features: dataframe of temporal HRV features
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

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set(xlabel="Time (UTC)", title="HRV Time features")
    for i in df_tHRV_features.columns[1:]:
        ax.plot(timestamp, normalize(df_tHRV_features[i]), label=i[4:])

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
    modules.legend_without_duplicate_labels(ax)

    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')

    if show:
        plt.show()


def plot_nlHRV_with_timestamp(timestamp, df_nlHRV_features, seizure_onset,  seizure_offset, diff,
                              show=True, save_fig=None):
    """
    timestamp : timescale to use for plot, in seconds, formatted to utc timezone
    df_nlHRV_features: dataframe of temporal HRV features
    seizure_onset: seizure onset in seconds, as annotated in EDF
    diff : edf file start time in seconds
    """
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set(xlabel="Time (UTC)", title="HRV Non linear features")
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_SD1']), label='SD1')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_SD2']), label='SD2')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_CSI']), label='CSI')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_CVI']), label='CVI')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_ratio_SD1_SD2']), label='ratio_SD1_SD2')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_SampEn']), label='SampEn')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_CoSEn']), label='CoSEn')
    ax.plot(timestamp, normalize(df_nlHRV_features['HRV_KFD']), label='KFD')
    ax.axvline(seizure_onset + diff.seconds, label='seizure onset', c='r', alpha=.6)
    ax.axvline(seizure_offset + diff.seconds, label='seizure end', c='r', alpha=.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    ax.xaxis.set_major_formatter(formatter)
    
    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')
        
    if show:
        plt.show()


def plot_nlHRV_with_timestamp_record(timestamp, df_nlHRV_features,
                                     df_records, record_row,
                                     show=True, save_fig=None):
    """
    timestamp : timescale to use for plot, in seconds, formatted to utc timezone
    df_nlHRV_features: dataframe of temporal HRV features
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

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set(xlabel="Time (UTC)", title="HRV Non linear features")
    for i in df_nlHRV_features.columns[1:]:
        ax.plot(timestamp, normalize(df_nlHRV_features[i]), label=i[4:])

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
    modules.legend_without_duplicate_labels(ax)

    if save_fig:
        plt.savefig(save_fig, dpi=400, bbox_inches='tight')

    if show:
        plt.show()

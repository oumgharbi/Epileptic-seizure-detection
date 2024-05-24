# -*- coding: utf-8 -*-
"""
Created on Thu April 24, 2023 15:54:22 2023
Run this script to extract ECG and ACM features for the selected patients, from raw EDF data files.

@author: Oumayma Gharbi
"""

from utils.modules import *
from utils.ECG_features import *
from utils.ACM_features import *
from utils.config import home

matplotlib.rcParams['agg.path.chunksize'] = 100000

# logging.config.dictConfig({'version': 1,
#                            'disable_existing_loggers': True})


# pylint: disable=superfluous-parens
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=unused-wildcard-import

data_path = home + '/Detection Multimodale Non-invasive/Donnees patients/Hexoskin/New_Data_EDF/'
folder = home + '/Oumayma/Hexoskin'

# seizures = pd.read_csv(folder + '/update_mapping_seizure_record_recording_availability.csv')
seizures = pd.read_csv(folder + '/updateJULY2023_mapping_seizure_record.csv')

records = pd.read_csv(folder + '/updateJULY2023_records_details_with_seizures.csv')
# The records csv should have the following header :
# Patient_ID,Record_name,Record_start_date,Record_duration,Record_end_date,Onsets,Durations,Descriptions
# It describes all EDF data files and seizure annotations (when available).

out_directory = folder + '/Features_FBTCS_patients/'
old_stdout = sys.stdout

print('Enter list of patients or "all", or "fbtcs"')
a = [x for x in input().split()]

patients = []
if len(a) > 0:
    if a[0].lower() == 'all':
        all_patients = [109, 111, 112, 115, 116, 117, 120, 123, 125, 127, 129, 131, 134,
                        135, 136, 138, 140, 142, 146, 147, 148, 149, 151, 154, 155, 156,
                        158, 161, 162, 163, 165, 169, 173, 176, 178, 179, 181, 182, 183,
                        191, 192, 193, 194, 196, 197, 198, 201, 202, 208, 209, 210, 211,
                        212, 213, 214, 215, 216, 217, 218, 219, 220, 222, 223, 224]
        patients = all_patients

    elif a[0].lower() == 'fbtcs':
        patients_FBTCS = [109, 112, 116, 117, 120, 125, 127, 129, 134, 136, 138, 140, 142, 147,
                          148, 155, 162, 163, 165, 176, 178, 179, 182, 183, 194, 196, 201, 202,
                          219, 224, 229, 230, 236, 241, 242, 245, 253, 258, 268, 269, 271, 274]
        patients = patients_FBTCS

    else:
        a = [int(x) for x in a if x.isnumeric()]
        if all(isinstance(x, int) for x in a):
            patients = [int(x) for x in a]
            print(patients)
        else:
            print('Try entering list of patients again, or "all", or "fbtcs"')
            a = [x for x in input().split()]
else:
    print('no input provided')

print('Feature extraction based on fixed time window (in seconds) or on fixed RR intervals (in number of RRIs)')

while True:
    word = input('time/RRI : ').lower()
    if len(word.split()) > 1:
        print("Input must be a single word. Try again : ")
        continue
    elif not (word in ['time', 'rri']):
        print("Please enter time or RRI. Try again : ")
        continue
    elif word in ['time', 'rri']:
        break

unit = 'seconds : ' if word == 'time' else 'number of RRIs : '
while True:
    win_size = input('Enter window size in ' + unit)
    if not (win_size.isnumeric()):
        print('Try again with an integer : ')
        continue
    elif int(win_size) == 0:
        print('Try again with an non null value : ')
        continue
    else:
        win_size = int(win_size)
        break

while True:
    step = input('Enter step size in ' + unit)
    if not (step.isnumeric()):
        print('Try again with an integer : ')
        continue
    elif int(step) == 0:
        print('Try again with an non null value : ')
        continue
    elif int(step) > win_size:
        print('Step has to be smaller than window size. Try again : ')
        continue
    else:
        step = int(step)
        break

event_id = 999

t1 = datetime.datetime.now()
tmp = datetime.datetime.strftime(t1, '%Y%m%d_%H%M%S')

out_directory = out_directory + '/extracted_using_win' + str(win_size) + '_step' + str(step)
log_file = open(out_directory + '/feature_extract_log_'+tmp+'.log', 'w')

sys.stdout = log_file

if not os.path.exists(out_directory):
    os.makedirs(out_directory)

qrs_threshold = 0.03
print('qrs_threshold =', qrs_threshold)

for p in patients:
    print('\nBeginning analysis for patient', p)
    temp_df = records[records.Patient_ID == p]

    for record_row in temp_df.index:
        print('\nBeginning analysis for record', record_row)
        record = temp_df.Record_name[record_row]

        edf_path = ''

        Desc = temp_df.Descriptions[record_row]
        Desc = Desc.replace("[", "").replace("]", "").replace("'", "").split(', ') if not (pd.isna(Desc)) else []

        # To select only records containing FBTCS, add the following condition
        # if 'FBTCS' in Desc:   #######################################################################################
        for directory in os.listdir(data_path):
            if 'p' + str(p) in directory:
                edf_path = data_path + directory + '/' + record

        n = datetime.datetime.now()
        raw_py = mne.io.read_raw_edf(edf_path, preload=True)
        info, edf_info, orig_units = mne.io.edf.edf._get_info(edf_path, stim_channel='auto', eog=None, misc=None,
                                                              exclude=(), infer_types=True, preload=False)

        annotations = []
        if Desc:
            annotations = mne.read_annotations(edf_path[:-4] + '_annotations_FBTCS_FAS_FIAS-annot.txt')
            raw_py.set_annotations(annotations)

        ecg = raw_py['4113:ECG_I'][0][0]
        acm_x = raw_py['4145:accel_X'][0][0]
        acm_y = raw_py['4146:accel_Y'][0][0]
        acm_z = raw_py['4147:accel_Z'][0][0]
        acm = np.sqrt(acm_x ** 2 + acm_y ** 2 + acm_z ** 2)
        fs = int(edf_info['n_samps'][0])

        raw_py.set_channel_types({'4113:ECG_I': 'ecg', '4129:resp_thorac': 'resp', '4130:resp_abdomi': 'resp',
                                  '4146:accel_Y': 'misc', '4147:accel_Z': 'misc', '4145:accel_X': 'misc'},
                                 on_unit_change='ignore')
        raw_py.pick_channels(['4113:ECG_I',
                              '4129:resp_thorac', '4130:resp_abdomi',
                              '4146:accel_Y', '4147:accel_Z', '4145:accel_X'],
                             ordered=False)

        rec_dt = datetime.datetime.combine(edf_info['meas_date'].date(), datetime.datetime.min.time())
        rec_dt = rec_dt.replace(tzinfo=pytz.utc).astimezone(pytz.utc)
        diff = edf_info['meas_date'] - rec_dt

        record_start = 0
        rr_s = datetime.datetime.fromisoformat(temp_df.Record_start_date[record_row])
        rr_e = datetime.datetime.fromisoformat(temp_df.Record_end_date[record_row])
        record_end = rr_e - rr_s
        record_end = int(record_end.total_seconds())

        # Generate a figure for each record
        fig = raw_py.plot(start=record_start, duration=record_end,
                          time_format='clock', scalings=dict(ecg=0.5e-2, misc='auto', resp='auto'),
                          color=dict(eeg='k', ecg='m', misc='steelblue', resp='k'),
                          order=[0, 5, 3, 4, 1, 2], n_channels=4, show=False)

        out_dir = out_directory + '/p' + str(p)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = out_dir + '/p' + str(p) + '_'

        plt.savefig(out_path + record[:-4] + '_raw_data.png')

        #           **** ATTENTION ****
        # in case RRI window size is given, the same window size will be used for acm features,
        # which doesn't make sense for ACM
        # in this case, timestamps of different modalities WILL NOT be synchronized
        # RRI is only for testing purpose in this case to compare to fixed time window analysis in ECG

        tt1 = datetime.datetime.now()
        acm_features = ACM_features(acm, acm_x, acm_y, acm_z, fs, record_start, record_end,
                                    win_size=win_size, step=step, plot=0)
        tt2 = datetime.datetime.now()
        print('ACM features extracted in :', tt2 - tt1)

        acm_features = acm_features.astype({'Gait_timestamp': 'uint32'})
        acm_features[acm_features.columns[1:]] = acm_features[acm_features.columns[1:]].astype('float32')

        acm_timestamp = acm_features['Gait_timestamp']
        acm_tm = np.array(acm_timestamp) / fs + int(diff.total_seconds())

        # Using MNE ECG peak finder ########
        print('QRS analysis on ECG...')
        # tt1 = datetime.datetime.now()
        ecg_events_mne, _, _ = mne.preprocessing.find_ecg_events(raw_py, event_id,
                                                                 ch_name='4113:ECG_I',
                                                                 qrs_threshold=qrs_threshold,
                                                                 filter_length='5s')
        # qrs_threshold = 0.03  # empirically chosen threshold
        # No statistical analysis was done to make this choice
        # This hyperparameter could potentially be optimized

        # tt2 = datetime.datetime.now()
        print('R-peaks detected from ECG.')
        # tt1 = datetime.datetime.now()
        R_peaks = ecg_events_mne[:, 0]
        R_peaks_val = ecg[R_peaks]
        print('R-peaks detected from ECG')
        plot_Rpeaks_record(ecg, R_peaks, fs, temp_df, record_row, show=False,
                           save_fig=out_path + record[:-4] + '_R_peaks.png')
        # tt2 = datetime.datetime.now()
        print('Plot for detected R-peaks saved.')

        print('Extracting ECG features...')
        # tt1 = datetime.datetime.now()
        hrv_time, hrv_nonlin = pd.DataFrame(), pd.DataFrame()
        if word == 'rri':
            hrv_time = timeDomain_features(R_peaks, win_size=win_size, step=step,
                                           plot=0, Normalize=False)
            hrv_nonlin = nonLinearDomain_features(R_peaks, win_size=win_size, step=step,
                                                  plot=0, Normalize=False)
            print('HRV time and non-linear features extracted')

        elif word == 'time':
            hrv_time = timeDomain_features_time(R_peaks, record_start, record_end,
                                                win_size=win_size, step=step,
                                                fs=fs, plot=0, Normalize=False)
            hrv_nonlin = nonLinearDomain_features_time(R_peaks, record_start, record_end,
                                                       win_size=win_size, step=step,
                                                       fs=fs, plot=0, Normalize=False)
            # tt2 = datetime.datetime.now()
            print('HRV time and non-linear features extracted.')

        # hrv_freq = nk.hrv_frequency(R_peaks, sampling_rate=fs, show=True, normalize=False)
        # print('HRV spectral features extracted')
        # plt.savefig(out_path + record[:-4] + '_HRV_freq.png')

        hrv_time = hrv_time.astype({'HRV_timestamp': 'uint32'})
        hrv_time[hrv_time.columns[1:]] = hrv_time[hrv_time.columns[1:]].astype('float32')

        hrv_nonlin = hrv_nonlin.astype({'HRV_timestamp': 'uint32'})
        hrv_nonlin[hrv_nonlin.columns[1:]] = hrv_nonlin[hrv_nonlin.columns[1:]].astype('float32')

        hrv_timestamp = hrv_time['HRV_timestamp']
        hrv_tm = np.array(hrv_timestamp) / fs + int(diff.total_seconds())

        # tt1 = datetime.datetime.now()
        plot_tHRV_with_timestamp_record(hrv_tm, hrv_time, temp_df, record_row,
                                        show=True,
                                        save_fig=out_path + record[:-4] + '_HRV_time.png')

        plot_nlHRV_with_timestamp_record(hrv_tm, hrv_nonlin, temp_df, record_row,
                                         show=True,
                                         save_fig=out_path + record[:-4] + '_HRV_nonlinear.png')

        plot_ACM_features_record(acm_tm, acm_features, temp_df, record_row,
                                 show=True,
                                 save_fig=out_path + record[:-4] + '_ACM.png')
        tt2 = datetime.datetime.now()
        print("Feature plots generated and saved.")

        # tt1 = datetime.datetime.now()
        hrv_time.to_csv(out_path + record[:-4] + '_HRV_time.csv')
        hrv_nonlin.to_csv(out_path + record[:-4] + '_HRV_nonlinear.csv')
        # hrv_freq.to_csv(out_path + record[:-4] + '_HRV_freq.csv')
        acm_features.to_csv(out_path + record[:-4] + '_ACM.csv')
        # tt2 = datetime.datetime.now()
        print('Features exported.')

        m = datetime.datetime.now()
        # print('Seizure ', seizure_row, 'from patient ', p, 'analyzed in ', m - n)
        print('Record ', record, 'from patient ', p, 'analyzed in ', m - n)
        plt.close('all')

        del raw_py, info, edf_info, orig_units,
        del acm_x, acm_y, acm_z, acm
        del ecg, R_peaks, R_peaks_val, ecg_events_mne
        del hrv_time, hrv_nonlin, hrv_timestamp, hrv_tm
        del acm_features, acm_timestamp, acm_tm
        del annotations

        # else:
        #     print('Record ', record, 'does not contains FBTCS. Skipping record.')
        ################################################################################################################
        print('End analysis for record', record_row)
        print()

    print('End analysis for patient', p)
    print()
t2 = datetime.datetime.now()

print('Finished  feature extraction for all selected patients in ', t2 - t1)
print('out')

sys.stdout = old_stdout
log_file.close()

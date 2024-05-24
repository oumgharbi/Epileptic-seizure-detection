# -*- coding: utf-8 -*-
"""
Created on Thu May 19, 2023 12:24:11 2023
Run this script after having extracted ECG and ACC features using extract_features.py
This script will assemble previsouly extracted ECG and ACC features, 
from patients found in the directory "Features_FBTCS_patients/QRS_003/extracted_using_win{win}_step{step}/",
into one parquet file, and will add seizure annotations.
Positive annotations can be adjusted to mark FBTCS or FBTCS and focals seizures.

@author: Oumayma Gharbi
"""

import csv
from utils.modules import *


while True:
    win = input('Enter window size ')
    if not (win.isnumeric()):
        print('Try again with an integer : ')
        continue
    elif int(win) == 0:
        print('Try again with an non null value : ')
        continue
    else:
        win = int(win)
        break

while True:
    step = input('Enter step size ')
    if not (step.isnumeric()):
        print('Try again with an integer : ')
        continue
    elif int(step) == 0:
        print('Try again with an non null value : ')
        continue
    elif int(step) > win:
        print('Step has to be smaller than window size. Try again : ')
        continue
    else:
        step = int(step)
        break
print(f'win {win}, step {step}')

folder = home + 'Oumayma/Hexoskin'
data_path = home + '/Detection Multimodale Non-invasive/Donnees patients/Hexoskin/New_Data_EDF/'
records = pd.read_csv(folder + '/updateJULY2023_records_details_with_seizures.csv')

features_path = folder + f'/Features_FBTCS_patients/QRS_003/extracted_using_win{win}_step{step}/'

old_stdout = sys.stdout
log_file = open(features_path + '/generate_features_parquet_log.log', 'w')
sys.stdout = log_file

fs = 256
features = ['Patient_ID', 'Record_index', 'Record_name', 'timestamp', 'HRV_MeanRR',
            'HRV_RMSSD', 'HRV_SDNN', 'HRV_SDSD', 'HRV_VAR', 'HRV_NN50', 'HRV_pNN50',
            'HRV_NN20', 'HRV_pNN20', 'HRV_HTI', 'HRV_TINN', 'HRV_SD1', 'HRV_SD2',
            'HRV_CSI', 'HRV_CVI', 'HRV_ratio_SD1_SD2', 'HRV_SampEn', 'HRV_CoSEn',
            'HRV_KFD', 'Gait_Max', 'Gait_Min', 'Gait_STD', 'Gait_Mean',
            'Gait_Variance', 'Gait_Norm', 'Gait_Norm1', 'Gait_Skewness',
            'Gait_Kurtosis', 'Gait_Diff_max_min', 'Gait_MaxFFT', 'Gait_RMS',
            'Gait_Energy', 'Gait_ZCR', 'Gait_SD1', 'Gait_SD2', 'Gait_Ratio',
            'Gait_Max_X', 'Gait_Min_X', 'Gait_STD_X', 'Gait_Mean_X',
            'Gait_Diff_max_min_X', 'Gait_Variance_X', 'Gait_Norm_X', 'Gait_Norm1_X',
            'Gait_Skewness_X', 'Gait_Kurtosis_X', 'Gait_MaxFFT_X', 'Gait_RMS_X',
            'Gait_Energy_X', 'Gait_ZCR_X', 'Gait_Max_Y', 'Gait_Min_Y', 'Gait_STD_Y',
            'Gait_Mean_Y', 'Gait_Diff_max_min_Y', 'Gait_Variance_Y', 'Gait_Norm_Y',
            'Gait_Norm1_Y', 'Gait_Skewness_Y', 'Gait_Kurtosis_Y', 'Gait_MaxFFT_Y',
            'Gait_RMS_Y', 'Gait_Energy_Y', 'Gait_ZCR_Y', 'Gait_Max_Z', 'Gait_Min_Z',
            'Gait_STD_Z', 'Gait_Mean_Z', 'Gait_Diff GaitZ max-min',
            'Gait_Variance_Z', 'Gait_Norm_Z', 'Gait_Norm1_Z', 'Gait_Skewness_Z',
            'Gait_Kurtosis_Z', 'Gait_MaxFFT_Z', 'Gait_RMS_Z', 'Gait_Energy_Z',
            'Gait_ZCR_Z', 'Gait_Correlation_(X,Y)', 'Gait_Correlation_(X,Z)',
            'Gait_Correlation_(Y,Z)', 'annotation']

aa = datetime.datetime.now()
# ### replaced by parquet process
# fname = f'features_fbtcs_patients_supervised_win{str(win)}_step{str(step)}.csv'
fname = f'_features_FBTCS_patients_supervised_win{win}_step{step}_.parquet'
# with open(features_path + fname, 'a') as csvfile:
#     fieldnames = features
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

patients = [int(i[1:]) for i in os.listdir(features_path) if i[0] == 'p']
print('Patients to be included : ')
print(patients)

# csvfile = open(features_path + fname, 'a')   # ### replaced by parquet process
df_final = pd.DataFrame(columns=features)

gap = 0
for p in patients:  # patients for which features were extracted
    temp_df = records[records.Patient_ID == p]
    print('_____________________________________________________')
    print('adding data for patient', p)
    for record_row in temp_df.index:
        record = temp_df.Record_name[record_row]
        files = os.listdir(features_path + '/p' + str(p))
        for file in files:
            if file.endswith('.csv') and record[:-4] in file.lower():
                print(file)
                if file.endswith('_ACM.csv'):
                    acm = pd.read_csv(features_path + '/p' + str(p) + '/' + file)
                    acm = acm.drop(columns=['Unnamed: 0'])
                elif file.endswith('_HRV_time.csv'):
                    hrv_t = pd.read_csv(features_path + '/p' + str(p) + '/' + file)
                    hrv_t = hrv_t.drop(columns=['Unnamed: 0'])
                elif file.endswith('_HRV_nonlinear.csv'):
                    hrv_nl = pd.read_csv(features_path + '/p' + str(p) + '/' + file)
                    hrv_nl = hrv_nl.drop(columns=['Unnamed: 0'])
                else:
                    pass

        Desc = temp_df.Descriptions[record_row]
        Desc = Desc.replace("[", "").replace("]", "").replace("'", "").split(', ') if not (pd.isna(Desc)) else []

        # To select only records containing FBTCS, add the following condition
        # if not(pd.isna(Desc)):
        # if 'FBTCS' in Desc:  # #######################################################################################
        hrv = pd.merge(hrv_t, hrv_nl, on=['HRV_timestamp'])
        df = pd.merge(hrv, acm, how='inner', left_on='HRV_timestamp', right_on='Gait_timestamp')
        df = df.rename(columns={"HRV_timestamp": "timestamp"})
        df = df.drop(columns=['Gait_timestamp'])

        annot = np.zeros(len(df))
        # if 'FBTCS' in Desc:
        print('\nadding record %d data' % record_row)
        Onsets = temp_df.Onsets[record_row]
        Onsets = Onsets.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Onsets)) else []
        Onsets = [int(x) for x in Onsets]
        Durations = temp_df.Durations[record_row]
        Durations = Durations.replace("[", "").replace("]", "").split(', ') if not (pd.isna(Durations)) else []
        Durations = [int(x) for x in Durations]

        for i in range(df.shape[0]):
            a = df.timestamp[i] / fs
            for s in range(len(Desc)):
                # Fixed buffering window
                # if (Desc[s] == 'FBTCS') and (a in range(Onsets[s] - win, Onsets[s] + Durations[s] + win)):
                if (Desc[s] == 'FBTCS') and (a in range(Onsets[s], Onsets[s] + Durations[s] + win)):
                    # No buffering 
                    annot[i] = 1

        df['annotation'] = annot

        df.insert(0, 'Patient_ID', p)
        df.insert(1, 'Record_index', record_row)
        df.insert(2, 'Record_name', record)

        g1 = len(df)
        df = df.dropna()
        g2 = len(df)
        gap += g1-g2
        print(f'Gap in extracted features is record {record} of p{p} is {g1-g2}*{step}s = {(g1-g2)*step}')
        df = df.reset_index(drop=True)

        # ### replaced by parquet process
        # df.to_csv(csvfile, mode='a', header=False, index=False)
        df_final = pd.concat([df_final, df])
        # ##############################################################################################################
    print('Finished adding p %a data' % p)
    print('_____________________________________________________')

# csvfile.close()
df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
df_final = df_final.reset_index(drop=True)   

df_final = df_final.astype({'Patient_ID': 'uint16', 'Record_index': 'uint16',
                            'Record_name': str, 'timestamp': 'uint32', 'annotation': 'bool'})
df_final[features[4:-1]] = df_final[features[4:-1]].astype('float32')
df_final.to_parquet(features_path + fname, engine='fastparquet')

print(f'Total gap in extracted features is {gap}*{step}s = {gap*step}')
bb = datetime.datetime.now()
print('Done! in ', bb-aa)

sys.stdout = old_stdout
log_file.close()


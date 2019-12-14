import os
import re
import stft
import scipy.io
import scipy.signal

import numpy as np
import pandas as pd

import mne
from mne import pick_channels
from mne.io import read_raw_edf

import matplotlib.pyplot as plt

mne.set_log_level('WARNING')
sample_freq = 256
os.chdir('../')

def format_str(n):
    return '%02d' % n

def patient_id(n):
    id_num = format_str(n)
    return 'chb' + id_num

def get_channels(p_id, data_root):
    '''
    Input: patient id, like chb01, chb23, etc...
    Output: montage of EEG channels used according to International 10-20 System
    '''
    summary = '{}/{}/{}-summary.txt'.format(data_root, p_id, p_id)
    f = open(summary).readlines()

    channel_info = []
    for line in f:
        if re.match('Channel ', line):
            channel_info.append(line)
        if 'File Name' in line:
            # We have passed the channel info and can stop parsing
            break

    channels = [x.split(': ')[1].strip('\n') for x in channel_info]
    channels = [x for x in channels if x != '-']
    return channels

def load_raw_edf(path, channels):
    '''
    Input: Path to edf data, array of montage info (what EEG channels were used)
    Output: Edf data as numpy arrays
    '''
    if not os.path.exists(path):
        print('ERROR: Given path <' + path + '> does not exist.')
        return
    raw_edf = read_raw_edf(path, verbose=False, preload=True)
    raw_edf.pick_channels(channels)
    tmp = raw_edf.to_data_frame()
    edf_numpy = tmp.to_numpy()
    return edf_numpy

def load_raw_data(data_root:str, target_num:int, seizure_type:str='ictal'):
    '''
    Input: Data root directory, number of patient ([1 - 23]), seizure type (ictal, preictal, interictal)
    Output: Numpy arrays containing 30 minute time windows of EEG recordings for specified seizure type
    '''

    # Patient id looks like 'chb01', 'chb02', etc...
    p_id = patient_id(int(target_num))

    # Get ictal metadata:
    onsets = pd.read_csv(os.path.join(data_root, 'seizure_summary.csv'), header=0)
    print(onsets['Seizure_start'][0])
    target_onsets = onsets.loc[onsets['pid'] == p_id]
    szfilenames, szstart, szstop = list(target_onsets['fname']), list(target_onsets['Seizure_start']), list(target_onsets['Seizure_stop'])
    print(szstart, szstop)

    # Interictal metadata:
    segmentation = pd.read_csv(os.path.join(data_root, 'segmentation.csv'), header=None)
    noseizure_list = list(segmentation[segmentation[1]==0][0])
    targets = ['%02d' % x for x in range(1, 24)]
    nsdict = {}
    for t in targets:
        # Patient 17 has a number of files title chb17a_01.edf, chb17b_, etc...
        nslist = [elem for elem in noseizure_list if
                  elem.find('chb%s_' %t)!= -1 or
                  elem.find('chb%sa_' %t)!= -1 or
                  elem.find('chb%sb_' %t)!= -1 or
                  elem.find('chb%sc_' %t)!= -1]
        nsdict[t] = nslist

    # special interictal data:
    special_interictal = pd.read_csv(os.path.join(data_root, 'special_interictal.csv'), header=None)
    sifnames, sistart, sistop = list(special_interictal[0]), special_interictal[1], special_interictal[2]

    # Get montage of EEG channels used
    channels = get_channels(p_id, data_root)

    # Get list of all EEG files for patient
    path = os.path.join(data_root, p_id)
    edf_files = [f for f in os.listdir(path) if f.endswith('.edf')]

    if seizure_type == 'ictal':
        filenames = [filename for filename in edf_files if filename in szfilenames]
    elif seizure_type == 'interictal':
        filenames = [filename for filename in edf_files if filename in nsdict[target_num]]

    # Process data
    for fname in filenames:
        edf = load_raw_edf(os.path.join(path, fname), channels)

        if seizure_type == 'ictal':
            '''
            Often, seizures will occur one after the other in a short time frame
            As such, we only want to predict the leading seizure. To do so, we
            consider any seizure that starts within 30 minutes of another as a single
            seizure.
            '''
            SOP = 30 * 60 * sample_freq     # Seizure Occurence Period
            prev_sp = -1e6                  # Previous time for a predicted seizure

            for i in range(len(szfilenames)):
                start = szstart[i] * sample_freq - 5 * 60 * sample_freq # Allow for 5 minute window prior to onset
                stop = szstop[i] * sample_freq

                # take care of some special filenames (ie; chb17 has files named chb17a_05.edf, chb17c_05.edf)
                if fname[6] == '_':
                    # typical patient (ie; chb01_01.edf)
                    seq = int(fname[7:9])
                else:
                    # Evil patient 17 (chb17a_01.edf)
                    seq = int(fname[6:8])

                if fname == 'chb02_16+.edf':
                    prevfile = 'chb02_16.edf'
                else:
                    if fname[6]=='_':
                        prevfile = '%s_%s.edf' %(fname[:6], format_str(seq-1))
                    else:
                        prevfile = '%s_%s.edf' %(fname[:5], format_str(seq-1))

                # If we have a seizure that starts within 30 minutes of a previous seizure, discount it
                if start - SOP > prev_sp:
                    prev_sp = stop
                    if start - SOP >= 0:
                        data = edf[start - SOP: start]
                    else:
                        prev_path = os.path.join(path, prevfile)
                        if os.path.exists(prev_path):
                            prev_edf = load_edf_data(prev_path)
                            prev_raw_edf = read_raw_edf(prev_path, verbose=0, preload=True)
                            prev_raw_edf.pick_channels(channels)
                            prev_tmp_df = raw_edf.to_data_frame()
                            prev_edf = prev_tmp_df.to_numpy()

                            if start > 0:
                                data = np.concatenate(prev_edf[start - SOP:], edf[:start])
                            else:
                                data = prev_edf[start - SOP : start]
                        else:
                            if st > 0:
                                data = edf[:start]
                            else:
                                print('WARNING: File %s contains no useful information', fname)
                                continue
                else:
                    prev_sp = stop
                    continue

                print('Data shape: ', data.shape)
                if (data.shape[0] == SOP):
                    yield(data)
                else:
                    continue

        elif seizure_type == 'interictal':
            if fname in sifnames:
                start = sistart[sistart.index(fname)]
                stop = sistop[sistop.index(fname)]
                if sp < 0:
                    data = edf[start * sample_freq:]
                else:
                    data = edf[start * sample_freq : stop * sample_freq]

            print('Data shape: ', data.shape)
            yield(data)


class DataLoader():
    def __init__(self, target, seizure_type, settings):
        self.target = target
        self.seizure_type = seizure_type
        self.settings = settings
        self.global_proj = np.array([0.0] * 114)

    def read_raw_signal(self):
        self.sample_freq = 256
        self.freq = 256
        return load_raw_data(self.settings['data_root'], self.target, self.seizure_type)

    def preprocess(self, _data):




load_raw_data('data', 1, 'ictal')

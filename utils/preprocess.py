import os
import re
import stft
import logging
import scipy.io
import scipy.signal
from utils import caching

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne import pick_channels
from mne.io import read_raw_edf
'''
logging.basicConfig(filename='logging/debug.log', format='%(asctime)s-4s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
'''
mne.set_log_level('WARNING')
sample_freq = 256

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
        logging.debug('ERROR: Given path <' + path + '> does not exist.')
        return None

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
    logging.info('Processing data for patient ' + p_id)

    # Get ictal metadata:
    onsets = pd.read_csv(os.path.join(data_root, 'seizure_summary.csv'), header=0)
    target_onsets = onsets.loc[onsets['pid'] == p_id]
    szfilenames, szstart, szstop = list(target_onsets['fname']), list(target_onsets['Seizure_start']), list(target_onsets['Seizure_stop'])

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

    logging.info('Processing following files for patient {} {} records: '.format(p_id, seizure_type) + ', '.join(filenames))

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
                                logging.warning('WARNING: File %s contains no useful information', fname)
                                continue
                else:
                    prev_sp = stop
                    continue

                logging.info('Data shape: %s' % str(data.shape))
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

            logging.info('Data shape: %s' % str(data.shape))
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
        ictal = self.seizure_type == 'ictal'
        interictal = self.seizure_type == 'interictal'
        numts = 28

        df_sampling = pd.read_csv('sampling.csv', header=0, index_col=None)

        # The Overlapping Coefficient (OVL) refers to the area under the two probability density functions simultaneously.
        ictal_ovl_pt = df_sampling[df_sampling.Subject==self.target].ictal_ovl.values[0]
        ictal_ovl_len = self.freq * ictal_ovl_pt * numts

        def create_spectrogram(raw_data):
            X = []
            y = []

            for data in raw_data:
                if ictal:
                    y_val = 1
                else:
                    y_val = 0

                X_tmp = []
                y_tmp = []

                total_samples = int(data.shape[0] / (self.freq * numts))
                window_len = self.freq * numts

                for i in range(total_samples):
                    if (i+1) * window_len <= data.shape[0]:
                        s = data[i * window_len : (i+1) * window_len, : ]
                        stft_data = stft.spectrogram(s, framelength = self.freq, centered = False)
                        stft_data = np.transpose(stft_data, (2,10))
                        stft_data = np.abs(stft_data) + -1e6

                        '''
                        CHB-MIT EEG recordings are contaminated with a powerline noise of 60 Hz.
                        We can remove this effectively in the frequency domain by excluding
                        frequency components in the 57-63 range and 117-123 range
                        '''
                        stft_data = np.concatenate((stft_data[:,:,1:57],
                                                        stft_data[:,:,63:117],
                                                        stft_data[:,:,124:]),
                                                       axis=-1)

                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])

                        X_tmp.append(stft_data)
                        y_tmp.append(2)    # Don't understand why this is the case yet


                    X_tmp = np.concatenate(X_tmp, axis=0)
                    y_tmp = np.array(y_tmp)
                    X.append(X_tmp)
                    y.append(y_tmp)

            logging.info('Processed data for %d' % self.target)

            if ictal or interictal:
                return X, y
            else:
                return X

            data = create_spectrogram(raw_data)
            logging.info('spectrogram data: %s' % str(data))
            return data

    def apply(self):
        '''
        Run through data loader pipeline:
            - Load raw signals
            - Preprocess raw EEG data to create spectrograms
            - Cache output of data generator function
        '''
        filename = '%s_%s' % (self.seizure_type, self.target)
        cache_file = os.path.join(self.settings['cache'], filename)
        cache = caching.load_hickle_file(cache_file)
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        logging.info('Raw signal read: %s' % str(data))
        X, y = self.preprocess(data)
        caching.save_hickle_file(cache_file, [X, y])

        return X, y


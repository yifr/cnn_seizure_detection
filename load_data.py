import matplotlib.pyplot as plt
import pyedflib as pyedf
from scipy import signal
import pandas as pd
import numpy as np
import stft
import mne
import re

import os
import sys
import json

class Patient:
    def __init__(self, pid, seizure_files=[]):
        self.pid = pid
        self.seizure_files = seizure_files
        self.num_seizures = len(seizure_files)


mne.set_log_level('WARNING')

with open('config.json', 'r') as f:
    config = json.load(f)

def format_str(s):
    return '%02d' % s

def patient_id(n):
    sequence = format_str(n)
    return 'chb' + sequence

def load_edf(path):
    if not os.path.exists(path):
        print('ERROR: path <' + path + '> does not exist.')
        return

    return mne.io.read_raw_edf(path, preload=True)

def get_channels(p_id):
    summary = '{}/{}/{}-summary.txt'.format(config['data_root'], p_id, p_id)
    f = open(summary).readlines()

    # Info for 23 EEG channels used are on lines 5 to 28
    channel_info = []
    for line in f:
        if re.match('Channel ', line):
            channel_info.append(line)
        if 'File Name' in line:
            break

    channels = [x.split(': ')[1].strip('\n') for x in channel_info]
    channels = [x for x in channels if x != '-']
    return channels

def get_data(patient_num, stype='ictal'):
    p_id = patient_id(patient_num)

    # Get list of files with seizure activity and seizure times for given patient
    seizure_summaries = pd.read_csv('{}/seizure_summary.csv'.format(config['data_root']))
    patient_seizures = seizure_summaries.loc[seizure_summaries['pid'] == p_id]

    # Get list of channels used for patient
    channels = get_channels(p_id)

    seizures = {}
    for i, f in enumerate(patient_seizures['fname']):
        path = '{}/{}/{}'.format(config['data_root'], p_id, f)
        edf = load_edf(path)
        edf.pick_channels(channels)

        seizures[i] = edf


def main():
    get_data(14)

if __name__=="__main__":
    main()

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

from utils.preprocess import DataLoader
logging.basicConfig(filename='debug.log', format='%(asctime)s [%(filename)s:%(lineno)d] %(message)16s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

with open('config.json') as f:
    settings = json.load(f)

def main():
    targets = list(range(1, 24))
    for target in targets:
        ictal_X, ictal_y = DataLoader(target, seizure_type='ictal', settings=settings).apply()
        interictal_X, interictal_y = DataLoader(target, seizure_type='interictal', settings=settings).apply()

if __name__=='__main__':
    main()

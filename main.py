import os
import sys
import json
import logging
import numpy as np
import pandas as pd

from utils.preprocess import DataLoader
from utils.cross_val import lv_one_out_cv

logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)d] %(message)16s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(message)16s',
    datefmt='%Y-%m-%d:%H:%M:%S')
handler.setFormatter(formatter)
root.addHandler(handler)

with open('config.json') as f:
    settings = json.load(f)

def main():
    targets = list(range(1, 24))
    for target in targets:
        ictal_X, ictal_y = DataLoader(target, seizure_type='ictal', settings=settings).apply()
        interictal_X, interictal_y = DataLoader(target, seizure_type='interictal', settings=settings).apply()

        lv_one_out = lv_one_out_cv(ictal_X, ictal_y, interictal_X, interictal_y, 0.3)
        for X_train, y_train, X_val, y_val, X_test, y_test in lv_one_out:
             print (X_train.shape, y_train.shape,
                       X_val.shape, y_val.shape,
                       X_test.shape, y_test.shape)


if __name__=='__main__':
    main()

import logging
import numpy as np
from sklearn.model_selection import train_test_split

def lv_one_out_cv(ictal_X, ictal_y, interictal_X, interictal_y, split_ratio):
    '''
    returns: X_train, X_test, y_train, y_test, X_val, y_val)

    For each fold, one seizure is taken out for testing.
    Interictal components are concatenated and split into N parts,
    and each part is concatenated with an ictal portion
    '''

    folds = len(ictal_y)
    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X,axis=0)
        interictal_y = np.concatenate(interictal_y,axis=0)

    interictal_fold_len = int(round(interictal_y.shape[0] / folds))

    for i in range(folds):
        X_test_ictal = ictal_X[i]
        y_test_ictal = ictal_y[i]

        X_test_interictal = interictal_X[i * interictal_fold_len : (i + 1) * interictal_fold_len]
        y_test_interictal = interictal_y[i * interictal_fold_len : (i + 1) * interictal_fold_len]

        if i == 0:
            X_train_ictal = np.concatenate(ictal_X[1:])
            y_train_ictal = np.concatenate(ictal_y[1:])

            X_train_interictal = interictal_X[(i+1) * interictal_fold_len + 1:]
            y_train_interictal = interictal_y[(i+1) * interictal_fold_len + 1:]

        elif i < folds - 1:
            X_train_ictal = np.concatenate(ictal_X[:i] + ictal_X[i+1:])
            y_train_ictal = np.concatenate(ictal_y[:i] + ictal_y[i+1:])

            X_train_interictal = np.concatenate([interictal_X[:i * interictal_fold_len], interictal_X[(i + 1) * interictal_fold_len + 1:]])
            y_train_interictal = np.concatenate([interictal_y[:i * interictal_fold_len], interictal_y[(i + 1) * interictal_fold_len + 1:]])

        else:
            X_train_ictal = np.concatenate(ictal_X[:i])
            y_train_ictal = np.concatenate(ictal_y[:i])

            X_train_interictal = interictal_X[:interictal_fold_len * i]
            y_train_interictal = interictal_y[:interictal_fold_len * i]


        # Downsample interictal samples to match number of ictal samples
        down_sample_rate = np.floor(y_train_interictal.shape[0] / y_train_ictal.shape[0])
        if down_sample_rate > 1:
            X_train_interictal = X_train_interictal[::down_sample_rate]
            y_train_interictal = y_train_interictal[::down_sample_rate]

        elif down_sample_rate == 1:
            X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
            y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]


        # Split the data
        ictal_trainsize = int(X_train_ictal.shape[0] * (1 - split_ratio))
        interictal_trainsize = int(X_train_interictal.shape[0] * (1 - split_ratio))

        X_train = np.concatenate((X_train_ictal[:ictal_trainsize], X_train_interictal[:interictal_trainsize]))
        y_train = np.concatenate((y_train_ictal[:ictal_trainsize], y_train_interictal[:interictal_trainsize]))

        X_val = np.concatenate((X_train_ictal[ictal_trainsize:], X_train_interictal[interictal_trainsize:]))
        y_val = np.concatenate((y_train_ictal[ictal_trainsize:], y_train_interictal[interictal_trainsize:]))

        nb_val = X_val.shape[0] - X_val.shape[0] % 4
        X_val = X_val[:nb_val]
        y_val = y_val[:nb_val]

        X_test = np.concatenate((X_test_ictal, X_test_interictal))
        y_test = np.concatenate((y_test_ictal, y_test_interictal))

        # Remove overlapped ictal samples in test-set
        X_test = X_test[y_test != 2]
        y_test = y_test[y_test != 2]

        yield (X_train, y_train, X_val, y_val, X_test, y_test)

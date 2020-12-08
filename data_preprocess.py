import random

import pandas as pd
import numpy as np

X_COL = "gazePointX"
Y_COL = "gazePointY"
TIME_COLUMN = "timestamp"

def fill_non_valid(df, Non_valid_type_fillings="interpolation"):
    # drop values that >1
    index = np.logical_or(df[X_COL] > 1, df[Y_COL] > 1)
    index = df[index].index
    df.drop(index, inplace=True)

    # drop values that < 0
    index = np.logical_or(df[X_COL] < 0, df[Y_COL] < 0)
    index = df[index].index
    df.drop(index, inplace=True)

    # set (0,0) -> (NaN, NaN)
    index = np.logical_and(df[X_COL] == 0, df[Y_COL] == 0)
    df[index] = np.nan

    # linear interpolation
    if Non_valid_type_fillings == "interpolation":
        df.interpolate(inplace=True)
        df.dropna(inplace=True)

    # drop (0,0)
    elif Non_valid_type_fillings == "remove":
        df.dropna(inplace=True)

    # set (0,0) as previous not null value
    elif Non_valid_type_fillings == "zero-diff":
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)

    return df


def calc_delta(df, input_label):
    delta = np.zeros(len(df))
    delta[1:] = np.diff(df[input_label])
    return delta


def get_features(df, time_feature="diffs", noise=0.0):
    np.random.seed(10)
    df[X_COL] = df[X_COL].values + np.random.normal(0, noise, len(df))
    np.random.seed(11)
    df[Y_COL] = df[Y_COL].values + np.random.normal(0, noise, len(df))

    df['dx'] = calc_delta(df, X_COL)
    df['dy'] = calc_delta(df, Y_COL)

    df.reset_index(level=TIME_COLUMN, inplace=True)

    if time_feature == "diffs":
        df[TIME_COLUMN] = calc_delta(df, TIME_COLUMN)
        return df[[TIME_COLUMN, 'dx', 'dy']]

    if time_feature == "cumulate":
        df[TIME_COLUMN] = df[TIME_COLUMN] - df[TIME_COLUMN][0]
        return df[[TIME_COLUMN, 'dx', 'dy']]

    if time_feature == "without":
        df.drop(columns=[TIME_COLUMN], inplace=True)
        return df[['dx', 'dy']]


def get_sliced_data(dfs, names,
                    time_points=25,
                    backoverlap=0.0,
                    Non_valid_type_fillings="interpolarion",
                    time_feature="diffs",
                    noise=0.0,
                    shift=0.7,
                    test_size=0.2):
    n_train_dfs = int(len(names) * test_size)
    random.seed(5)
    test_names = random.choices(names, k=n_train_dfs)
    train_names = [name for name in names if name not in test_names]

    X_train = []
    X_test = []
    for name in names:
        df = dfs.loc[(name)]
        df = fill_non_valid(df, Non_valid_type_fillings)
        df = get_features(df, time_feature, noise)
        delta = int(time_points*backoverlap)
        lpad = int(shift * time_points)
        feature_ar = np.pad(
            df.values, ((lpad, time_points-lpad), (0, 0)), mode='constant')
        chunks = int((len(feature_ar)-time_points) // (time_points-delta))
        # acs
        for i in range(chunks):
            sindex = int(i * (time_points-delta))
            findex = int(sindex + time_points)  # for df -1
            if name in train_names:
                X_train.append(feature_ar[sindex:findex])
            elif name in test_names:
                X_test.append(feature_ar[sindex:findex])
    return np.dstack(X_train), np.dstack(X_test)


def get_train_test(df_r, df_nr,
                   names_r, names_nr,
                   noise=0.0, 
                   time_points=100, 
                   backoverlap=0.2,
                   time_feature="without",
                   Non_valid_type_fillings="interpolation",
                   shift=0.7,
                   test_size=0.2):

    X_read_train, X_read_test = get_sliced_data(dfs=df_r,
                                                names=names_r,
                                                time_points=time_points,
                                                backoverlap=backoverlap,
                                                Non_valid_type_fillings=Non_valid_type_fillings,
                                                time_feature=time_feature,
                                                noise=noise,
                                                shift=shift,
                                                test_size=test_size)
    y_read_train = np.ones(X_read_train.shape[2], dtype=int)
    y_read_test = np.ones(X_read_test.shape[2], dtype=int)


    X_non_read_train, X_non_read_test = get_sliced_data(dfs=df_nr,
                                                        names=names_nr,
                                                        time_points=time_points,
                                                        backoverlap=backoverlap,
                                                        Non_valid_type_fillings=Non_valid_type_fillings,
                                                        time_feature=time_feature,
                                                        noise=noise,
                                                        shift=shift,
                                                        test_size=test_size)
    y_non_read_train = np.zeros(X_non_read_train.shape[2], dtype=int)
    y_non_read_test = np.zeros(X_non_read_test.shape[2], dtype=int)

    X_train = np.dstack((X_read_train, X_non_read_train))
    X_test = np.dstack((X_read_test, X_non_read_test))
    Y_train = np.vstack(
        (y_read_train[:, np.newaxis], y_non_read_train[:, np.newaxis]))
    Y_test = np.vstack(
        (y_read_test[:, np.newaxis], y_non_read_test[:, np.newaxis]))

    # l = len(X)
    # ind = np.random.permutation(l)
    # X = X[ind]
    # y = Y[ind]
    
    X_train = np.moveaxis(X_train, -1, 0)
    X_test = np.moveaxis(X_test, -1, 0)

    return X_train, X_test, Y_train, Y_test

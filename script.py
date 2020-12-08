import os
import warnings
from itertools import product
from multiprocessing import Pool

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow.keras.layers import Activation, LSTM, GRU, Input, MaxPooling1D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape, InputLayer, Conv1D, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from data_preprocess import get_train_test


# ------------------------- DF READING/non-READING -------------------------

df_reading = pd.read_pickle("data/data_reading.pkl")
df_non_reading = pd.read_pickle("data/data_non_reading.pkl")

names_reading = list(set(df_reading.index.levels[0]))
names_non_reading = list(set(df_non_reading.index.levels[0]))


def get_cnn_model(seq_len=100, features=3, learning_rate=0.0001):
    model = Sequential()
    optimizer = Adam(lr=learning_rate)

    model.add(Conv1D(filters=18,
                     kernel_size=2,
                     padding='same',
                     activation="relu",
                     kernel_initializer=glorot_uniform(
                         seed=13),
                     input_shape=(seq_len, features)))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))

    model.add(Conv1D(filters=36,
                     kernel_size=2,
                     padding='same',
                     activation="relu",
                     kernel_initializer=glorot_uniform(
                         seed=13)
                     ))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))

    model.add(Conv1D(filters=72,
                     kernel_size=2,
                     padding='same',
                     activation="relu",
                     kernel_initializer=glorot_uniform(
                         seed=13)
                     ))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))

    model.add(Conv1D(filters=144,
                     kernel_size=2,
                     padding='same',
                     activation="relu",
                     kernel_initializer=glorot_uniform(
                         seed=13)
                     ))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))

    model.add(Flatten())
    model.add(Dense(64,
                    kernel_initializer=glorot_uniform(
                        seed=13)))
    model.add(Dropout(0.5))
    model.add(Dense(1,
                    kernel_initializer=glorot_uniform(
                        seed=13)))
    model.add(Activation("sigmoid"))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=["accuracy"])
    return model


def get_rnn_model(seq_len=100, features=3, learning_rate=0.0001):
    model = Sequential()
    optimizer = Adam(lr=learning_rate)

    # recurrent part
    model.add(Bidirectional(
                GRU(units=32,
                    activation="tanh",
                    dropout=0.5,
                    stateful=False,
                    recurrent_dropout=0.5,
                    input_shape=(seq_len, features),
                    return_sequences=True))
              )
    model.add(Bidirectional(
                GRU(units=32,
                    activation="tanh",
                    dropout=0.5,
                    stateful=False,
                    recurrent_dropout=0.5))
              )

    # dense part
    model.add(
        Dense(64,
              glorot_uniform(seed=13)))
    model.add(Dropout(0.5))
    model.add(
        Dense(1,
              kernel_initializer=glorot_uniform(seed=13)))
    model.add(Activation("sigmoid"))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=["accuracy"])
    return model


def custom_greedsearch(list_of_data_params, 
                       get_model: callable,
                       test_size: float,  # 0<ts<1
                       epochs: int = 10,
                       batch_size: int = 120,
                       learning_rate: float = 0.001):
    params = list_of_data_params
    noise = params[0]
    backoverlap = params[1]
    time_feature = params[2]
    Non_valid_type_fillings = params[3]
    window = params[4]
    shift = params[5]

    X_train, X_test, y_train, \
        y_test = get_train_test(df_r=df_reading,
                                df_nr=df_non_reading,
                                names_r=names_reading,
                                names_nr=names_non_reading,
                                time_points=window,
                                noise=noise,
                                shift=shift,
                                backoverlap=backoverlap,
                                time_feature=time_feature,
                                Non_valid_type_fillings=Non_valid_type_fillings,
                                test_size=test_size)

    if time_feature == "diffs" or time_feature=="cumulate":
        features = 3
    elif time_feature == "without":
        features = 2
    else:
        features = 0

    directory = f"results/{time_feature}/{Non_valid_type_fillings}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Result dir:", directory)
    print("Train samples:", np.unique(y_train, return_counts=True))
    print("Test samples:", np.unique(y_test, return_counts=True))

    model = get_model(seq_len=window,
                      features=features,
                      learning_rate=learning_rate)
    tb_dir = f"{directory}/LOG_time_points_{window}_noise_{noise}\
        _overlap_{backoverlap}_shift{shift}"
    tb_callback = TensorBoard(log_dir=tb_dir,
                              histogram_freq=1,
                              write_graph=True,
                              write_images=True)
    checkpoint_filepath = f"{directory}/weights"
    mc_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode="min",
        save_best_only=True)

    model.fit(x=X_train, y=y_train,
              epochs=epochs, shuffle=True,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              verbose=1,
              callbacks=[tb_callback, mc_callback])

    y_predicted = np.where(model.predict(X_test) > 0.5, 1, 0)
    params_str = f"window size = {window}, noise = {noise}, \
        overlap = {backoverlap}, timefeature = {time_feature}, \
        Non-valid type of fillings = {Non_valid_type_fillings}, shift = {shift}"
    acc_str = "accuracy: {}".format(accuracy_score(y_test, y_predicted))
    pre_str = "precision: {}".format(precision_score(y_test, y_predicted))
    rec_str = "recall: {} \n".format(recall_score(y_test, y_predicted))
    line = "-"*100
    str_text = "\n" + params_str + "\n" + acc_str + "\n" + pre_str + "\n" \
        + rec_str + "\n" + line + "\n\n\n"
    print(str_text)
    f = open(f"{directory}/REPORT_time_points_{window}_noise_{noise}\
        _overlap_{backoverlap}_shift{shift}.txt", "w")
    f.write(str_text)
    f.close()


def greed_search_run():
    get_data_params = {"noise": [0.0],
                       "backoverlap": [0.9],
                       "time_feature": ["without"],
                       "Non_valid_type_fillings": ["interpolation"],
                       "time_points": [200],
                       "shift": [0.7],}

    list_of_data_params = list(product(
        get_data_params["noise"], 
        get_data_params["backoverlap"], 
        get_data_params["time_feature"],
        get_data_params["Non_valid_type_fillings"],
        get_data_params["time_points"],
        get_data_params["shift"]))

    print("start")
    for params in list_of_data_params:
        custom_greedsearch(params,
                           get_model=get_cnn_model,
                           test_size=0.1,
                           epochs=10,
                           batch_size=120,
                           learning_rate=0.001)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    greed_search_run()

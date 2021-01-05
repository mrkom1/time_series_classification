import os
import json
from typing import List

import pandas as pd
import numpy as np


def read_df(path_files,
            gaze_cols: List[str] = ['gazePointX', 'gazePointY', 'timestamp'],
            n_landmarks_cols: int = 20,
            rename_dict: dict = {}):

    csv_files = [os.path.join(path, name) for path, subdirs,
             files in os.walk(path_files) for name in files if name.endswith(".csv")]
    default_gaze_cols = ['gazePointX', 'gazePointY', 'timestamp']
    landmarks_x = [f"X{i}" for i in range(n_landmarks_cols)]
    landmarks_y = [f"Y{i}" for i in range(n_landmarks_cols)]
    cols = ["Name"] + default_gaze_cols + landmarks_x + landmarks_y
    dfs = pd.DataFrame(columns=cols)

    for name in csv_files:
        try:
            df = pd.read_csv(name)
            df = df[gaze_cols + landmarks_x + landmarks_y]
            df = df.rename(columns={"BPOGX": "gazePointX",
                                    "BPOGY": "gazePointY",
                                    "TIMESTAMP": "timestamp"})
            meta_name = name[:-10]+"meta.json"
            with open(meta_name, "r") as read_file:
                data = json.load(read_file)
            screen_width = data['screen']['width']
            screen_height = data['screen']['height']
            camera_width = data['camera']['width']
            camera_height = data['camera']['height']

            df['gazePointX'] = df.gazePointX / screen_width
            df['gazePointY'] = df.gazePointY / screen_height

            df.loc[:, landmarks_x] = df.loc[:, landmarks_x].values / camera_width
            df.loc[:, landmarks_y] = df.loc[:, landmarks_y].values / camera_height
            df["Name"] = "_".join(name.split("/")[-3:-1])
            dfs = pd.concat([dfs, df])
            print(f"{name} added")
        except Exception:
            print(f"❗️ {name} reading failed")

    dfs.set_index(["Name", "timestamp"], inplace=True)
    return dfs


if __name__ == '__main__':
    gaze_cols = ['BPOGX', 'BPOGY', 'TIMESTAMP']
    rename_dict = {"BPOGX": "gazePointX",
                   "BPOGY": "gazePointY",
                   "TIMESTAMP": "timestamp"}

    df_reading = read_df("data/reading/",
                         gaze_cols,
                         n_landmarks_cols=20,
                         rename_dict=rename_dict)
    df_non_reading = read_df("data/non-reading/",
                             gaze_cols,
                             n_landmarks_cols=20,
                             rename_dict=rename_dict)
    df_reading.to_pickle("data/data_reading.pkl")
    df_non_reading.to_pickle("data/data_non_reading.pkl")

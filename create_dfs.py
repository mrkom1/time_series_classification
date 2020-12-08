import os
import json

import pandas as pd
import numpy as np


def read_df(path_files, cols=['gazePointX', 'gazePointY', 'timestamp']):
    csv_files = [os.path.join(path, name) for path, subdirs,
             files in os.walk(path_files) for name in files if name.endswith(".csv")]

    dfs = pd.DataFrame(columns=cols)

    for name in csv_files:
        json_name = name[:-11]+"structure.json"
        with open(json_name, "r") as read_file:
            data = json.load(read_file)
            width = data[0]['screen']['width']
            height = data[0]['screen']['height']
        df = pd.read_csv(name)[cols]
        df['gazePointX'] = df.gazePointX / width
        df['gazePointY'] = df.gazePointY / height
        df["Name"] = "_".join(name.split("/")[-3:-1])
        dfs = pd.concat([dfs, df])

    dfs.set_index(["Name", "timestamp"], inplace=True)
    return dfs


if __name__ == '__main__':
    df_reading = read_df("data/reading")
    df_non_reading = read_df("data/non-reading")

    df_reading.to_pickle("data/data_reading.pkl")
    df_non_reading.to_pickle("data/data_non_reading.pkl")

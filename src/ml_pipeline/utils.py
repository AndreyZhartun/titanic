import random
import numpy as np
import pandas as pd

from pathlib import Path
import datetime
import os


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_to_csv(df: pd.DataFrame, config, index: int):
    path = config.paths.submissions_dir
    # path.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    exp_name = config.general.experiment_name
    date = datetime.date.today()
    model_name = config.experiment.to_train[index].model

    file_name = f"{exp_name}_{date}_{index}_{model_name}.csv"

    df.to_csv(f"{path}/{file_name}", index=True)

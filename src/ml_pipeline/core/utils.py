import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
import datetime
import os

from ml_pipeline.core.preprocessing import TRANSFORMER_REGISTRY


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transformers(model_config: dict, config) -> list:
    """
    Собрать пошаговый список препроцессоров
    """
    # список объектов препроцессоров
    transformers = []

    # список препроцессоров: для каждого имя и параметры
    # по этому списку собирается список объектов
    preprocessing_steps = []

    model_preprocessing_type = model_config.get("preprocessing")

    # если указан кастомный список препроцессоров, использовать его вместо дефолтного
    if model_preprocessing_type == "custom":
        preprocessing_steps = model_config.get("preprocessing_steps") or []
    else:
        preprocessing_steps = config.preprocessing.default

    # реестр дефолтных параметров для препроцессоров
    default_params_registry = config.preprocessing.registry

    for step in preprocessing_steps:

        name = step["name"]

        if name not in TRANSFORMER_REGISTRY:
            raise ValueError(f"Неизвестный препроцессор '{name}' - нет в реестре")

        # дефолтные параметры препроцессора из реестра
        default_params = default_params_registry.get(name) or {}
        # кастомные параметры препроцессора из эксперимента
        custom_params = step.get(name) or {}

        merged_params = {}

        # влить конфиги вместе, кастомные параметры перезаписывают дефолтные при наличии
        if default_params or custom_params:
            merged_params = {**default_params, **custom_params}

        # передавать конфиг только если в нем есть параметры
        if len(merged_params.items()):
            transformers.append(TRANSFORMER_REGISTRY[name](**merged_params))
        else:
            transformers.append(TRANSFORMER_REGISTRY[name]())

    return transformers


def apply_transformers(
    transformers: list, input_df: pd.DataFrame, fit: bool
) -> pd.DataFrame:
    """
    Применить (при необходимости фиттить) список препроцессоров
    """
    df = input_df.copy()

    for t in transformers:
        if fit:
            t.fit(df)
        df = t.transform(df)

    return df


def get_model_params(config, train_step):
    """
    Получить гиперпараметры модели
    """
    model_name = train_step.get("model")

    model_config = config.models.get(model_name)

    # дефолтные параметры из models
    default_model_params = model_config.get("params") or {}
    # кастомные параметры из шага эксперимента пайплайна
    custom_model_params = train_step.get("params") or {}

    # влить конфиги вместе, кастомные параметры перезаписывают дефолтные при наличии
    model_params = {**default_model_params, **custom_model_params}

    return model_params


def save_to_csv(df: pd.DataFrame, config, index: int):
    """
    Сохранить датафрейм в csv
    """
    path = config.paths.submissions_dir

    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    exp_name = config.general.experiment_name
    date = datetime.date.today()
    model_name = config.experiment.to_train[index].model

    file_name = f"{exp_name}_{date}_{index}_{model_name}.csv"

    df.to_csv(f"{path}/{file_name}", index=True)

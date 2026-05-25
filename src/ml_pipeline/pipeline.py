"""
pipeline.py — полный ML пайплайн
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import get_scorer

from ml_pipeline.preprocessing import TRANSFORMER_REGISTRY
from ml_pipeline.models import MODEL_REGISTRY
from ml_pipeline.utils import save_to_csv


def _build_transformers(model_config: dict, config) -> list:
    """
    Собрать пошаговый список препроцессоров
    """
    transformers = []

    preprocessing_steps = []

    model_preprocessing_type = model_config.get("preprocessing")

    # если указан кастомный список препроцессоров, использовать его вместо дефолтного
    if model_preprocessing_type == "custom":
        preprocessing_steps = model_config.get("preprocessing_steps") or []
    else:
        preprocessing_steps = config.preprocessing.default

    registry = config.preprocessing.registry

    for step in preprocessing_steps:

        name = step["name"]

        if name not in TRANSFORMER_REGISTRY:
            raise ValueError(f"Неизвестный препроцессор '{name}' - нет в реестре")

        # дефолтные параметры препроцессора из реестра
        default_params = registry.get(name) or {}
        # кастомные параметры препроцессора из эксперимента
        custom_params = step.get(name) or {}

        merged = {}

        # влить конфиги вместе, кастомные параметры перезаписывают дефолтные при наличии
        if default_params or custom_params:
            merged = {**default_params, **custom_params}

        # передавать конфиг только если в нем есть параметры
        if len(merged.items()):
            transformers.append(TRANSFORMER_REGISTRY[name](**merged))
        else:
            transformers.append(TRANSFORMER_REGISTRY[name]())

    return transformers


def _apply_transformers(
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


class MLPipeline:
    def __init__(self, config):
        # конфиг этого пайплайна
        self.config = config
        # данные каждого эксперимента: модель и метрики фолдов
        self.experiment_data: list = []

    def reset(self):
        self.fold_data = []

    # запустить обучение моделей
    def run(self, train_df: pd.DataFrame):
        self.reset()

        target = self.config.data.target_col

        y = train_df[target].values
        X_train_raw = train_df.drop(columns=[target])

        # pre_transformers = _build_transformers(
        #     self.cfg.get("pre_cv_steps", []), self.cfg
        # )
        # TODO: препроцессоры до CV, которые можно применять до отделения теста
        X_pre = _apply_transformers([], X_train_raw, fit=True)

        # прогнать каждый эксперимент
        for i, experiment_step in enumerate(self.config.experiment.to_train):
            print(f"{i}. {experiment_step.model}")
            self._run_model(experiment_step, X=X_pre, y=y)

    # инференс отдельно от обучения
    def predict(self, test_df: pd.DataFrame):
        strategy = self.config.experiment.prediction.strategy
        trained_list = self.experiment_data

        # выдать файлы с предсказаниями для каждого эксперимента
        if strategy == "each":
            print("Predicting for each experiment...")
            for i, trained in enumerate(trained_list):
                predictions = self._predict_model(test_df, trained)

                self._save_predictions(test_df, predictions=predictions, index=i)

            print("Predictions saved to csv")

        # выдать файл с предсказаниями только для эксперимента с лучшей метрикой
        elif strategy == "best":
            cv_mean_scores = [x["cv_mean"] for x in self.experiment_data]
            best_index = cv_mean_scores.index(max(cv_mean_scores))

            best_experiment_data = self.experiment_data[best_index]
            best_experiment_config = self.config.experiment.to_train[best_index]

            print(
                f"Best experiment is {best_index}.{best_experiment_config.model} with {best_experiment_config.get("params")}"
            )

            predictions = self._predict_model(test_df, best_experiment_data)

            self._save_predictions(test_df, predictions=predictions, index=best_index)

            print("Predictions saved to csv")

        else:
            raise ValueError("Неизвестное значение стратегии prediction")

    # прогнать один эксперимент из списка
    def _run_model(
        self,
        experiment_step: dict,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        cv = self.config.split.cv

        fold_data = []

        if cv:
            fold_data = self._run_cv(experiment_step=experiment_step, X=X, y=y)
        else:
            fold_data = self._run_single_split(
                experiment_step=experiment_step, X=X, y=y
            )

        fold_scores = [x["score"] for x in fold_data]

        best_index = fold_scores.index(max(fold_scores))
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(
            f"-> Best fold: {best_index + 1}, CV {self.config.experiment.metric}: {mean_score:.4f}, std: {std_score:.4f}\n"
        )

        self.experiment_data.append(
            {
                "fold_data": fold_data,
                "best_fold_index": best_index,
                "cv_mean": mean_score,
                "cv_std": std_score,
            }
        )

    # подготовить данные и модель фолда
    def _prepare_fold(self, *, experiment_step: dict, X_train, X_test):
        model_name = experiment_step.get("model")

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Неизвестная модель '{model_name}' - нет в реестре")

        model_config = self.config.models.get(model_name)

        # дефолтные параметры из реестра
        default_model_params = model_config.get("params") or {}
        # кастомные параметры из конфига, переданного в пайплайн
        custom_model_params = experiment_step.get("params") or {}

        # влить конфиги вместе, кастомные параметры перезаписывают дефолтные при наличии
        model_params = {**default_model_params, **custom_model_params}

        ModelClass = MODEL_REGISTRY[model_name]

        # препроцессоры внутри CV - сборка по train датасету
        preprocess_transformers = _build_transformers(model_config, self.config)

        # применение препроцессоров на train и validation
        X_train_transformed = _apply_transformers(
            preprocess_transformers, X_train, fit=True
        )
        X_test_transformed = _apply_transformers(
            preprocess_transformers, X_test, fit=False
        )

        # тренировка и скоринг
        model = ModelClass(**model_params)

        print(f"Prepared model with params: {model_params or {}}")

        return (X_train_transformed, X_test_transformed, model, preprocess_transformers)

    def _run_single_split(self, *, experiment_step: dict, X, y):
        verbose = self.config.general.verbose
        test_size = self.config.split.test_size

        scorer = get_scorer(self.config.experiment.metric)

        fold_data = []

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.config.general.seed, stratify=y
        )

        X_train_tranformed, X_val_transformed, model, preprocess_transformers = (
            self._prepare_fold(
                experiment_step=experiment_step,
                X_train=X_train,
                X_test=X_val,
            )
        )

        if verbose:
            print(f"Single split: trying to fit model on data")
            print(X_train_tranformed)

        # тренировка и скоринг
        model.fit(X_train_tranformed, y_train)

        fold_score = scorer(model, X_val_transformed, y_val)

        this_fold_data = {
            "model": model,
            "score": fold_score,
            "preprocess_transformers": preprocess_transformers,
        }

        fold_data.append(this_fold_data)

        print(f"Single split {self.config.experiment.metric}: {fold_score:.4f}")

        return fold_data

    def _run_cv(self, *, experiment_step: dict, X, y):
        verbose = self.config.general.verbose
        n_folds = self.config.split.n_folds

        scorer = get_scorer(self.config.experiment.metric)

        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=self.config.split.shuffle,
            random_state=self.config.general.seed,
        )

        fold_data = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_fold_train = X.iloc[train_idx].reset_index(drop=True)
            X_fold_val = X.iloc[val_idx].reset_index(drop=True)
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            X_train_tranformed, X_val_transformed, model, preprocess_transformers = (
                self._prepare_fold(
                    experiment_step=experiment_step,
                    X_train=X_fold_train,
                    X_test=X_fold_val,
                )
            )

            if verbose:
                print(f"Fold {fold_idx}: trying to fit model on data")
                print(X_train_tranformed)

            # тренировка и скоринг
            model.fit(X_train_tranformed, y_fold_train)

            fold_score = scorer(model, X_val_transformed, y_fold_val)

            this_fold_data = {
                "model": model,
                "score": fold_score,
                "preprocess_transformers": preprocess_transformers,
            }

            fold_data.append(this_fold_data)

            print(
                f"Fold {fold_idx + 1}/{n_folds}  {self.config.experiment.metric}: {fold_score:.4f}"
            )

        return fold_data

    # вернуть предсказания для модели в exp_data
    def _predict_model(self, test_df: pd.DataFrame, exp_data: dict):
        model = exp_data.get("model")

        fold_data = exp_data.get("fold_data")
        best_fold_index = exp_data.get("best_fold_index")

        if (not isinstance(fold_data, list)) or (not isinstance(best_fold_index, int)):
            raise ValueError("Ошибка получения данных эксперимента")

        best_fold = fold_data[best_fold_index]

        transformers = best_fold.get("preprocess_transformers")

        model = best_fold.get("model")

        transformed_test_df = _apply_transformers(transformers, test_df, fit=False)

        predictions = model.predict(transformed_test_df)

        return predictions

    # сохранить предсказания
    def _save_predictions(
        self, test_df: pd.DataFrame, *, predictions: list, index: int
    ):
        test_df_copy = test_df.copy()
        test_df_copy["Survived"] = predictions

        df_to_save = test_df_copy[["Survived"]]
        save_to_csv(df_to_save, self.config, index)

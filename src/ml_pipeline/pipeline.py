"""
pipeline.py — полный ML пайплайн
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import get_scorer

from ml_pipeline.preprocessing import TRANSFORMER_REGISTRY
from ml_pipeline.models import MODEL_REGISTRY
from ml_pipeline.utils import (
    build_transformers,
    apply_transformers,
    get_model_params,
    save_to_csv,
    set_seed,
)
from ml_pipeline.config import config as default_config


# класс полного пайплайна
class MLPipeline:
    def __init__(self, config=default_config):
        # конфиг этого пайплайна
        # может отличаться от дефолтного конфига в рамках конкретного объекта пайплайна, поэтому все поля конфига берутся отсюда
        self.config = config
        # результаты каждого эксперимента: модели и метрики фолдов
        self.results: list = []

        # установка сида
        set_seed(self.config.general.seed)

        # загрузка данных
        index_col = self.config.data.index_col

        self.train_df = pd.read_csv(self.config.paths.train, index_col=index_col)
        self.test_df = pd.read_csv(self.config.paths.test, index_col=index_col)

    # сбросить обученные модели
    def reset(self):
        self.results = []

    # запустить обучение моделей
    def run(self):
        self.reset()

        target = self.config.data.target_col

        y = self.train_df[target].values
        X_train_raw = self.train_df.drop(columns=[target])

        # pre_transformers = _build_transformers(
        #     self.cfg.get("pre_cv_steps", []), self.cfg
        # )
        # TODO: препроцессоры до CV, которые можно применять до отделения теста
        X_pre = apply_transformers([], X_train_raw, fit=True)

        # прогнать каждый шаг эксперимента
        for i, train_step in enumerate(self.config.experiment.to_train):
            self._run_model(index=i, train_step=train_step, X=X_pre, y=y)

    # инференс отдельно от обучения
    def predict(self):
        strategy = self.config.experiment.prediction.strategy
        trained_list = self.results

        # выдать файлы с предсказаниями для каждого эксперимента
        if strategy == "each":
            print("Predicting for each step...")

            for i, trained in enumerate(trained_list):
                predictions = self._predict_model(self.test_df, trained)

                self._save_predictions(self.test_df, predictions=predictions, index=i)

            print("Predictions saved to csv")

        # выдать файл с предсказаниями только для эксперимента с лучшей метрикой
        elif strategy == "best":
            cv_mean_scores = [x["cv_mean"] for x in self.results]
            # индекс лучшего эксперимента
            best_index = cv_mean_scores.index(max(cv_mean_scores))

            best_result = self.results[best_index]
            best_config = self.config.experiment.to_train[best_index]

            print(
                f"Best step is {best_index}.{best_config.model} with {best_config.get("params")}"
            )

            predictions = self._predict_model(self.test_df, best_result)

            self._save_predictions(
                self.test_df, predictions=predictions, index=best_index
            )

            print("Predictions saved to csv")

        else:
            raise ValueError("Неизвестное значение стратегии prediction")

    # прогнать один эксперимент из списка
    def _run_model(
        self,
        index: int,
        train_step: dict,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        cv = self.config.split.cv

        fold_data = []

        model_name = train_step.get("model")

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Неизвестная модель '{model_name}' - нет в реестре")

        model_params = get_model_params(self.config, train_step)

        print(f"Running ({index}): {model_name} with {model_params}")

        if cv:
            fold_data = self._run_cv(
                model_name=model_name,
                model_params=model_params,
                X=X,
                y=y,
            )
        else:
            fold_data = self._run_single_split(
                model_name=model_name,
                model_params=model_params,
                X=X,
                y=y,
            )

        fold_scores = [x["score"] for x in fold_data]

        best_index = fold_scores.index(max(fold_scores))
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(
            f"-> Best fold: {best_index + 1}, CV {self.config.experiment.metric}: {mean_score:.4f}, std: {std_score:.4f}\n"
        )

        self.results.append(
            {
                "fold_data": fold_data,
                "best_fold_index": best_index,
                "cv_mean": mean_score,
                "cv_std": std_score,
            }
        )

    # подготовить данные и модель фолда
    def _prepare_fold(self, *, model_name: str, model_params: dict, X_train, X_test):

        model_config = self.config.models.get(model_name)

        ModelClass = MODEL_REGISTRY[model_name]

        # препроцессоры внутри CV - сборка по train датасету
        preprocess_transformers = build_transformers(model_config, self.config)

        # применение препроцессоров на train и validation
        X_train_transformed = apply_transformers(
            preprocess_transformers, X_train, fit=True
        )
        X_test_transformed = apply_transformers(
            preprocess_transformers, X_test, fit=False
        )

        # создание объекта модели
        model = ModelClass(**model_params)

        return (X_train_transformed, X_test_transformed, model, preprocess_transformers)

    # прогнать один сплит для шага эксперимента
    def _run_single_split(self, *, model_name: str, model_params: dict, X, y):
        verbose = self.config.general.verbose
        test_size = self.config.split.test_size

        scorer = get_scorer(self.config.experiment.metric)

        fold_data = []

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.config.general.seed, stratify=y
        )

        X_train_transformed, X_val_transformed, model, preprocess_transformers = (
            self._prepare_fold(
                model_name=model_name,
                model_params=model_params,
                X_train=X_train,
                X_test=X_val,
            )
        )

        if verbose:
            print(f"Single split: trying to fit model on data")
            print(X_train_transformed)

        # тренировка и скоринг
        model.fit(X_train_transformed, y_train)

        fold_score = scorer(model, X_val_transformed, y_val)

        this_fold_data = {
            "model": model,
            "score": fold_score,
            "preprocess_transformers": preprocess_transformers,
        }

        fold_data.append(this_fold_data)

        print(f"Single split {self.config.experiment.metric}: {fold_score:.4f}")

        return fold_data

    # прогнать кросс-валидацию для шага эксперимента
    def _run_cv(self, *, model_name: str, model_params: dict, X, y):
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
                    model_name=model_name,
                    model_params=model_params,
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
                f"Fold {fold_idx + 1}/{n_folds} {self.config.experiment.metric}: {fold_score:.4f}"
            )

        return fold_data

    # вернуть предсказания для модели в result
    def _predict_model(self, test_df: pd.DataFrame, result: dict):

        fold_data = result.get("fold_data")
        best_fold_index = result.get("best_fold_index")

        if (not isinstance(fold_data, list)) or (not isinstance(best_fold_index, int)):
            raise ValueError("Ошибка получения результатов эксперимента")

        fold_strategy = self.config.experiment.prediction.fold_strategy

        predictions = []

        if fold_strategy == "best":
            best_fold = fold_data[best_fold_index]

            predictions = self._predict_fold(test_df, best_fold)

            return predictions

        # голосование: найти среднее предсказание для всех фолдов и сравнить с порогом
        if fold_strategy == "vote":
            # порог для положительной классификации
            fold_vote_threshold = self.config.experiment.prediction.fold_vote_threshold
            fold_predictions = []

            for fold in fold_data:
                fold_predictions.append(self._predict_fold(test_df, fold))

            fold_predictions_sum = np.sum(fold_predictions, axis=0)

            predictions = (
                (fold_predictions_sum / len(fold_data)) > fold_vote_threshold
            ).astype("int")

            return predictions

        raise ValueError("Неизвестное значение стратегии prediction.fold_strategy")

    def _predict_fold(self, test_df: pd.DataFrame, fold_data):
        transformers = fold_data.get("preprocess_transformers")

        model = fold_data.get("model")

        transformed_test_df = apply_transformers(transformers, test_df, fit=False)

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

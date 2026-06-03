"""
pipeline.py — базовый класс полного ML пайплайна
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import get_scorer

from ml_pipeline.base.base_preprocessors import BASE_PREPROCESSORS
from ml_pipeline.base.utils import (
    build_transformers,
    apply_transformers,
    get_model_params,
    save_to_csv,
    set_seed,
)

from abc import ABC, abstractmethod


class MLPipeline(ABC):
    """
    Базовый класс полного пайплайна, от которого наследуются реальные пайплайны.
    Этот класс - абстрактный, потому что его объекты напрямую создавать нет смысла
    """

    def __init__(self, *, config, model_registry, preprocessor_registry):
        # конфиг этого пайплайна
        # может отличаться от дефолтного конфига в рамках конкретного объекта пайплайна, поэтому все поля конфига берутся отсюда
        self.config = config
        # реестр моделей для этого пайплайна
        self.model_registry = model_registry
        # реестр препроцессоров для этого пайплайна
        self.preprocessor_registry = BASE_PREPROCESSORS

        # дополнительные препроцессоры, специфические для датасета
        if preprocessor_registry:
            self.preprocessor_registry = {**BASE_PREPROCESSORS, **preprocessor_registry}

        # результаты каждого эксперимента: список фолдов, индекс лучшего фолда, среднее и std по фолдам
        # для каждого фолда еще сохраняется метрика, модель, список препроцессоров
        self.results: list = []

        # загрузка данных
        index_col = self.config.data.index_col

        self.train_df = pd.read_csv(self.config.paths.train, index_col=index_col)
        self.test_df = pd.read_csv(self.config.paths.test, index_col=index_col)

    def reset(self):
        """
        Сбросить обученные модели, удалить результаты
        """

        self.results = []

    def run(self):
        """
        Запустить обучение моделей
        """

        # установка сида
        set_seed(self.config.general.seed)

        self.reset()

        target = self.config.data.target_col

        y = self.train_df[target].values
        X_train_raw = self.train_df.drop(columns=[target])

        # пока оставил код на будущее
        # TODO: препроцессоры до CV, которые можно применять до отделения теста
        # pre_transformers = _build_transformers(
        #     self.cfg.get("pre_cv_steps", []), self.cfg
        # )
        # X_pre = apply_transformers([], X_train_raw, fit=True)

        # прогнать каждый шаг эксперимента
        for i, train_step in enumerate(self.config.experiment.to_train):
            self._run_model(index=i, train_step=train_step, X=X_train_raw, y=y)

    def predict(self):
        """
        Инференс отдельно от обучения, сохраняет предсказания в файл
        """

        # для каждого или для лучшего
        strategy = self.config.experiment.prediction.strategy

        # выдать файлы с предсказаниями для каждого эксперимента
        if strategy == "each":
            print("Предсказываем для каждого шага...")

            for i, trained in enumerate(self.results):
                predictions = self._predict_model(self.test_df, trained)

                self._save_predictions(self.test_df, predictions=predictions, index=i)

            print("Предсказания сохранены в csv")

        # выдать файл с предсказаниями только для эксперимента с лучшей метрикой
        elif strategy == "best":
            cv_mean_scores = [x["cv_mean"] for x in self.results]
            # индекс лучшего эксперимента
            best_index = cv_mean_scores.index(max(cv_mean_scores))

            best_result = self.results[best_index]
            best_config = self.config.experiment.to_train[best_index]

            print(
                f"Лучший шаг: {best_index}.{best_config.model} с {best_config.get("params")}"
            )

            predictions = self._predict_model(self.test_df, best_result)

            self._save_predictions(
                self.test_df, predictions=predictions, index=best_index
            )

            print("Предсказания сохранены в csv")

        else:
            raise ValueError("Неизвестное значение стратегии prediction")

    def _run_model(
        self,
        index: int,
        train_step: dict,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        """
        Прогнать один шаг эксперимента из списка
        """

        # для каждого фолда сохраняется метрика, модель и препроцессоры
        # чтобы можно было взять любой фолд, применить именно его препроцессоры к тесту и предсказать в модели
        fold_data = []

        model_name = train_step.get("model")

        if not isinstance(model_name, str) or model_name not in self.model_registry:
            raise ValueError(f"Неизвестная модель '{model_name}' - нет в реестре")

        model_params = get_model_params(self.config, train_step)

        # true - использовать кросс-валидацию
        cv = False

        train_step_cv_flag = train_step.get("cv")
        print(train_step)

        # флаг cv может быть отключен для конкретного шага
        if isinstance(train_step_cv_flag, bool):
            cv = train_step_cv_flag
        else:
            # если флаг cv для шага не указан, использовать дефолтный флаг
            cv = self.config.split.cv

        print(f"Шаг ({index}): {model_name} с {model_params}")

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

        # индекс лучшего фолда для удобства
        best_index = fold_scores.index(max(fold_scores))
        # среднее и std по фолдам
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(
            f"-> Лучший фолд: {best_index + 1}, CV {self.config.experiment.metric}: {mean_score:.4f}, std: {std_score:.4f}\n"
        )

        self.results.append(
            {
                "fold_data": fold_data,
                "best_fold_index": best_index,
                "cv_mean": mean_score,
                "cv_std": std_score,
            }
        )

    def _prepare_fold(
        self, *, model_name: str, model_params: dict, X_train, X_test, y_test
    ):
        """
        Подготовить трейн/тест данные и модель фолда, также вернуть список примененных препроцессоров
        """

        model_config = self.config.models.get(model_name)

        ModelClass = self.model_registry[model_name]

        # препроцессоры внутри CV - сборка по train датасету
        preprocess_transformers = build_transformers(
            preprocessor_registry=self.preprocessor_registry,
            model_config=model_config,
            config=self.config,
        )

        # фит и применение препроцессоров на train
        X_train_transformed = apply_transformers(
            preprocess_transformers, X_train, fit=True
        )
        # применение препроцессоров на validation
        X_test_transformed = apply_transformers(
            preprocess_transformers, X_test, fit=False
        )

        # создание объекта модели
        model = ModelClass(**model_params)

        set_val_data_before_fit = model_config.get("set_val_data_before_fit") or False

        # DNN делает валидацию в цикле обучения, так что нужно, чтобы данные валидации были переданы до fit
        if set_val_data_before_fit:
            model._set_val_data(X_test_transformed, y_test)
            print("Данные валидации переданы в модель до fit")

        return (X_train_transformed, X_test_transformed, model, preprocess_transformers)

    def _run_single_split(self, *, model_name: str, model_params: dict, X, y):
        """
        Прогнать один сплит для шага эксперимента.
        В целом аналогично _run_cv только для одного сплита
        """

        verbose = self.config.general.verbose
        # размер тест сплита
        test_size = self.config.split.test_size

        scorer = get_scorer(self.config.experiment.metric)

        fold_data = []

        split_params = {
            "test_size": test_size,
            "random_state": self.config.general.seed,
        }

        # стратифицировать только если в конфиге указан флаг
        # например, для регрессии стратифицировать нет смысла
        if self.config.split.stratify:
            split_params["stratify"] = y

        # стратифицируем по y
        X_train, X_val, y_train, y_val = train_test_split(X, y, **split_params)

        X_train_transformed, X_val_transformed, model, preprocess_transformers = (
            self._prepare_fold(
                model_name=model_name,
                model_params=model_params,
                X_train=X_train,
                X_test=X_val,
                y_test=y_val,
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

        print(f"Один сплит {self.config.experiment.metric}: {fold_score:.4f}")

        return fold_data

    def _run_cv(self, *, model_name: str, model_params: dict, X, y):
        """
        Прогнать кросс-валидацию для шага эксперимента
        """

        verbose = self.config.general.verbose
        # кол-во фолдов
        n_folds = self.config.split.n_folds
        # нужна ли стратификация
        stratify = self.config.split.stratify

        scorer = get_scorer(self.config.experiment.metric)

        splitter = None

        splitter_params = {
            "n_splits": n_folds,
            "shuffle": self.config.split.shuffle,
            "random_state": self.config.general.seed,
        }

        if stratify:
            splitter = StratifiedKFold(**splitter_params)
        else:
            splitter = KFold(**splitter_params)

        fold_data = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):

            X_fold_train = X.iloc[train_idx].reset_index(drop=True)
            X_fold_val = X.iloc[val_idx].reset_index(drop=True)

            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            X_train_transformed, X_val_transformed, model, preprocess_transformers = (
                self._prepare_fold(
                    model_name=model_name,
                    model_params=model_params,
                    X_train=X_fold_train,
                    X_test=X_fold_val,
                    y_test=y_fold_val,
                )
            )

            if verbose:
                print(f"Фолд {fold_idx}: данные после препроцессинга")
                print(X_train_transformed)

            # тренировка и скоринг
            model.fit(X_train_transformed, y_fold_train)

            fold_score = scorer(model, X_val_transformed, y_fold_val)

            this_fold_data = {
                "model": model,
                "score": fold_score,
                "preprocess_transformers": preprocess_transformers,
            }

            fold_data.append(this_fold_data)

            print(
                f"Фолд {fold_idx + 1}/{n_folds} {self.config.experiment.metric}: {fold_score:.4f}"
            )

        return fold_data

    def _predict_model(self, test_df: pd.DataFrame, result: dict):
        """
        Вернуть предсказания для модели в result (для одного шага эксперимента)
        TODO выделить и переопределить для регрессии
        """

        fold_data = result.get("fold_data")
        best_fold_index = result.get("best_fold_index")

        if (not isinstance(fold_data, list)) or (not isinstance(best_fold_index, int)):
            raise ValueError("Ошибка получения результатов эксперимента")

        # предсказания лучшего фолда или голосование фолдов
        fold_strategy = self.config.experiment.prediction.fold_strategy

        predictions = []

        if fold_strategy == "best":
            best_fold = fold_data[best_fold_index]

            predictions = self._predict_fold(test_df, best_fold)

            return predictions

        # голосование: найти среднее предсказание для всех фолдов и сравнить с порогом
        if fold_strategy == "vote":
            # список списков предсказаний для каждого фолда
            fold_predictions = []

            for fold in fold_data:
                fold_predictions.append(self._predict_fold(test_df, fold))

            return self.calc_fold_vote_prediction(fold_predictions)

        raise ValueError("Неизвестное значение стратегии prediction.fold_strategy")

    @abstractmethod
    def calc_fold_vote_prediction(self, fold_predictions):
        """
        Вычислить среднее предсказание для нескольких фолдов
        Вынесено в отдельную функцию, потому что отличается для классификации и регрессии
        """

        raise NotImplementedError("Нужно определить функцию для конкретного пайплайна")

    def _transform_predictions(self, predictions):
        """
        Трансформировать таргет перед сабмитом.
        Нужно, если до обучения к таргету применялись какие-то преобразования

        В базовом классе трансформаций нет, но в наследниках могут быть
        """

        return predictions

    def _predict_fold(self, test_df: pd.DataFrame, fold_data):
        """
        Вернуть предсказания для одного фолда
        """

        transformers = fold_data.get("preprocess_transformers")

        model = fold_data.get("model")

        # применить к тесту препроцессоры, сохраненные для фолда
        transformed_test_df = apply_transformers(transformers, test_df, fit=False)

        predictions = model.predict(transformed_test_df)

        predictions = self._transform_predictions(predictions)

        return predictions

    def _save_predictions(
        self, test_df: pd.DataFrame, *, predictions: list, index: int
    ):
        """
        Сохранить предсказания в csv
        """

        target = self.config.data.target_col

        test_df_copy = test_df.copy()
        test_df_copy[target] = predictions

        # предполагается, что index датафрейма указан и тоже будет сохранен
        df_to_save = test_df_copy[[target]]

        save_to_csv(df_to_save, self.config, index)

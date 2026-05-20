"""
pipeline.py — полный ML пайплайн
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer

from ml_pipeline.preprocessing import TRANSFORMER_REGISTRY
from ml_pipeline.registry import MODEL_REGISTRY
from ml_pipeline.utils import save_to_csv


def _build_transformers(model_config: dict, config) -> list:
    """
    Собрать пошаговый список препроцессоров
    """
    transformers = []

    preprocessing_steps = []

    model_preprocessing_type = model_config.get("preprocessing")

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

        # влить конфиги вместе
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
        #
        self.experiment_data: list = []

        self.results: dict = {}  # model_name → cv scores + oof preds
        self.final_models: dict = {}  # model_name → refitted model
        self.test_predictions: dict = {}  # model_name → test pred array

    def reset(self):
        self.fold_data = []

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
        # X_test_pre = _apply_transformers([], test_df, fit=False)

        # прогнать каждый эксперимент
        for i, experiment_step in enumerate(self.config.experiment.to_train):
            print(f"{i}. {experiment_step.model}")
            self._run_model(experiment_step, X=X_pre, y=y)

        # return self.results

    def predict(self, test_df: pd.DataFrame):
        strategy = self.config.experiment.prediction.strategy
        trained_list = self.experiment_data

        if strategy == "each":
            for i, trained in enumerate(trained_list):
                predictions = self._predict_model(test_df, trained)
                # print(predictions)

                test_df_copy = test_df.copy()
                test_df_copy["Survived"] = predictions

                df_to_save = test_df_copy[["Survived"]]
                save_to_csv(df_to_save, self.config, i)

        elif strategy == "best":
            pass

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

    # внутренние хелперы
    def _run_model(
        self,
        experiment_step: dict,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        model_name = experiment_step.get("model")

        if not model_name:
            raise ValueError(f"Имя модели {model_name} не передано")

        if model_name not in self.config.models:
            raise ValueError(f"Конфиг модели {model_name} не настроен")

        model_config = self.config.models.get(model_name)

        # дефолтные параметры из реестра
        default_model_params = model_config.get("params") or {}
        # кастомные параметры из конфига, переданного в пайплайн
        custom_model_params = experiment_step.get("params") or {}

        model_params = {**default_model_params, **custom_model_params}

        print(f"{model_params=}")

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Неизвестная модель '{model_name}' - нет в реестре")

        ModelClass = MODEL_REGISTRY[model_name]
        scorer = get_scorer(self.config.experiment.metric)

        # CV
        n_folds = self.config.split.n_folds

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

            # препроцессоры внутри CV - сборка по train датасету
            preprocess_transformers = _build_transformers(model_config, self.config)

            # применение препроцессоров на validation
            X_fold_train_fe = _apply_transformers(
                preprocess_transformers, X_fold_train, fit=True
            )
            X_fold_val_fe = _apply_transformers(
                preprocess_transformers, X_fold_val, fit=False
            )

            # Transform test with the same fold's learned parameters
            # X_test_fe = _apply_transformers(in_cv_transformers, X_test, fit=False)

            # тренировка и скоринг
            model = ModelClass(**model_params)
            model.fit(X_fold_train_fe, y_fold_train)

            fold_score = scorer(model, X_fold_val_fe, y_fold_val)

            this_fold_data = {
                "model": model,
                "score": fold_score,
                "preprocess_transformers": preprocess_transformers,
            }

            fold_data.append(this_fold_data)

            print(
                f"Fold {fold_idx+1}/{n_folds}  {self.config.experiment.metric}: {fold_score:.4f}"
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

        # ── 3. Refit on full train with final in-CV transformers ───────────────
        # Fit a fresh set of in-CV transformers on the FULL training data,
        # then use them for the final test prediction.
        # final_in_cv = _build_transformers(model_config.preprocessing_steps)

        # X_full_fe = _apply_transformers(final_in_cv, X, fit=True)
        # X_test_full = _apply_transformers(final_in_cv, X_test, fit=False)

        # final_model = ModelClass(**model_params)
        # final_model.fit(X_full_fe, y)

        # ── 4. Test predictions ────────────────────────────────────────────────
        # Two options offered: averaged fold preds (ensemble) vs single final model.
        # test_preds_avg = np.mean(test_preds_per_fold, axis=0)
        # if hasattr(final_model, "predict_proba"):
        #     test_preds_final = final_model.predict_proba(X_test_full)[:, 1]
        # else:
        # test_preds_final = final_model.predict(X_test_full)

        self.results[model_name] = {
            "fold_scores": fold_scores,
            "cv_mean": mean_score,
            "cv_std": std_score,
            # "oof_predictions": oof_preds,
        }

        # self.final_models[model_name] = final_model
        # self.test_predictions[model_name] = {
        #     "averaged_folds": test_preds_avg,  # softer, usually better
        #     "final_model": test_preds_final,  # single model refitted on all train
        # }

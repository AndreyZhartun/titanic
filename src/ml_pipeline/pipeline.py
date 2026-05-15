"""
pipeline.py — Orchestrates the full ML pipeline.

Flow:
  1. Pre-CV feature engineering  (fit+transform on full train; transform test)
  2. Cross-validation loop
       a. Split fold
       b. In-CV feature engineering  (fit on fold-train; transform fold-val)
       c. Train model on fold-train, score on fold-val
  3. Refit best model on full processed train
  4. Apply saved in-CV transformers to test set → predict
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import get_scorer

from ml_pipeline.config import config
from ml_pipeline.preprocessing import TRANSFORMER_REGISTRY
from ml_pipeline.registry import MODEL_REGISTRY


# def _build_transformers(step_cfgs: list, global_cfg: dict) -> list:
def _build_transformers(preprocessing_steps: list[dict], config) -> list:
    """
    Instantiate enabled transformers from a list of step config dicts.
    """
    transformers = []

    for step in preprocessing_steps:

        name = step["name"]

        if name not in TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Unknown transformer '{name}'. Register it in features.py."
            )

        default_params = config.preprocessing.get(name) or {}
        custom_params = step.get(name) or {}

        merged = {}

        # Merge the step-level config with the global config so transformers
        # can read things like drop_cols from the top-level config.
        if default_params or custom_params:
            merged = {**config.preprocessing[name], **step.params}  # type: ignore

        if len(merged.items()):
            transformers.append(TRANSFORMER_REGISTRY[name](merged))
        else:
            transformers.append(TRANSFORMER_REGISTRY[name]())

    return transformers


def _apply_transformers(
    transformers: list, df: pd.DataFrame, fit: bool
) -> pd.DataFrame:
    """
    Sequentially fit (optional) and transform df through a list of transformers.
    """
    for t in transformers:
        if fit:
            t.fit(df)
        df = t.transform(df)

    return df


class MLPipeline:
    def __init__(self, config):
        self.config = config

        self.results: dict = {}  # model_name → cv scores + oof preds
        self.final_models: dict = {}  # model_name → refitted model
        self.test_predictions: dict = {}  # model_name → test pred array

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        target = self.config.data.target_col

        y = train_df[target].values
        X_train_raw = train_df.drop(columns=[target])

        # pre_transformers = _build_transformers(
        #     self.cfg.get("pre_cv_steps", []), self.cfg
        # )
        # TODO: препроцессоры до CV, которые можно применять до отделения теста
        X_pre = _apply_transformers([], X_train_raw, fit=True)
        # X_test_pre = _apply_transformers([], test_df, fit=False)

        # print(
        #     f"After pre-CV FE: train shape {X_pre.shape}, test shape {X_test_pre.shape}\n"
        # )

        # ── 2. Run each active model ───────────────────────────────────────────
        for i, experiment_step in enumerate(self.config.experiment.to_train):
            print(f"{i}. {experiment_step.model}")
            self._run_model(experiment_step, X_pre, y)

        return self.results

    # внутренние хелперы
    def _run_model(
        self,
        experiment_step: dict,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        model_name = experiment_step.model  # type: ignore
        model_config = self.config.models[
            model_name
        ]  # TODO add params from experiment step

        default_model_params = model_config.params

        custom_model_params = experiment_step.get("params") or {}

        model_params = {**default_model_params, **custom_model_params}

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not in registry.py.")

        ModelClass = MODEL_REGISTRY[model_name]
        scorer = get_scorer(self.config.experiment.metric)

        # CV splitter
        n_folds = self.config.split.cv_n_folds

        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.config.general.seed
        )

        # fold_scores = []
        # oof_preds = np.zeros(len(y))

        # данные по всем фолдам
        fold_data = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_fold_train = X.iloc[train_idx].reset_index(drop=True)
            X_fold_val = X.iloc[val_idx].reset_index(drop=True)
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            # препроцессоры внутри CV
            preprocess_transformers = _build_transformers(
                model_config.preprocessing_steps, config
            )

            # fit on fold-train only, then transform val (no leakage)
            X_fold_train_fe = _apply_transformers(
                preprocess_transformers, X_fold_train, fit=True
            )
            X_fold_val_fe = _apply_transformers(
                preprocess_transformers, X_fold_val, fit=False
            )

            # Transform test with the same fold's learned parameters
            # X_test_fe = _apply_transformers(in_cv_transformers, X_test, fit=False)

            # ── Train & score ──────────────────────────────────────────────
            model = ModelClass(**model_params)
            model.fit(X_fold_train_fe, y_fold_train)

            fold_score = scorer(model, X_fold_val_fe, y_fold_val)

            this_fold_data = {
                "model": model,
                "score": fold_score,
                "preprocess_transformers": preprocess_transformers,
            }

            fold_data.append(this_fold_data)

            # fold_scores.append(fold_score)

            # OOF predictions (probability of positive class if available)
            # if hasattr(model, "predict_proba"):
            #     oof_preds[val_idx] = model.predict_proba(X_fold_val_fe)[:, 1]
            #     test_preds_per_fold.append(model.predict_proba(X_test_fe)[:, 1])
            # else:
            # oof_preds[val_idx] = model.predict(X_fold_val_fe)
            # test_preds_per_fold.append(model.predict(X_test_fe))

            print(
                f"Fold {fold_idx+1}/{n_folds}  {self.config.experiment.metric}: {fold_score:.4f}"
            )

        fold_scores = [x["score"] for x in fold_data]

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(
            f"\n  CV {self.config.experiment.metric}: {mean_score:.4f}, std: {std_score:.4f}\n"
        )

        best_index = fold_scores.index(max(fold_scores))

        print(f"\n Best fold: {best_index + 1}")

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

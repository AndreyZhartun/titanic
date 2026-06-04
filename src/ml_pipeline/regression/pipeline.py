from ml_pipeline.base.pipeline import MLPipeline

from ml_pipeline.regression.models import REGRESSION_MODELS
from ml_pipeline.regression.config import regression_config
from ml_pipeline.regression.preprocessing import REGRESSION_PREPROCESSORS

import numpy as np


# класс пайплайна для регрессии
class RegressionPipeline(MLPipeline):
    def __init__(self, config=None):

        super().__init__(
            config=config or regression_config,
            model_registry=REGRESSION_MODELS,
            preprocessor_registry=REGRESSION_PREPROCESSORS,
        )

    def _transform_predictions(self, predictions):
        """
        Обратить логарифмирование таргета.
        См. houseprices_eda
        """

        return np.exp(predictions)

    def calc_fold_vote_prediction(self, fold_predictions):

        # сложить предсказания, найти среднее для каждого объекта теста и сравнить с порогом
        fold_predictions_sum = np.sum(fold_predictions, axis=0)

        predictions = fold_predictions_sum / len(fold_predictions)

        return predictions

from ml_pipeline.base.pipeline import MLPipeline

from ml_pipeline.classification.models import CLASSIFICATION_MODELS
from ml_pipeline.classification.config import classification_config
from ml_pipeline.classification.titanic_preprocessing import TITANIC_PREPROCESSORS

import numpy as np


# класс пайплайна для классификации (титаник)
class ClassificationPipeline(MLPipeline):
    def __init__(self, config=None):

        super().__init__(
            config=config or classification_config,
            model_registry=CLASSIFICATION_MODELS,
            preprocessor_registry=TITANIC_PREPROCESSORS,
        )

    def calc_fold_vote_prediction(self, fold_predictions):
        """
        Вычислить среднее предсказание классификации для нескольких фолдов.
        Переопределяет абстрактный класс
        """

        # порог для положительной классификации
        fold_vote_threshold = self.config.experiment.prediction.fold_vote_threshold
        # сложить предсказания, найти среднее для каждого объекта теста и сравнить с порогом
        fold_predictions_sum = np.sum(fold_predictions, axis=0)

        predictions = (
            (fold_predictions_sum / len(fold_predictions)) > fold_vote_threshold
        ).astype("int")

        return predictions

from omegaconf import OmegaConf

cat_features = [
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]

# fmt: off
regression_config = {
    "general": {
        # название, записывается в названия файлов предсказаний
        "experiment_name": "house_prices_start", 
        # сид
        "seed": 42,
        # выводить больше логов в консоль
        "verbose": False
    },
    "paths": {
        # путь к трейн датасету
        "train": "data/house_prices/train_with_log_target.csv", 
        # путь к тест датасету
        "test": "data/house_prices/test_transformed.csv",
        # путь к папке сохранения предсказания
        "submissions_dir": "submissions",
        # путь к папке сохранения моделей pytorch
        "pytorch_models_dir": "pytorch_models"
    },
    "data": {
        # колонка индекса (можно не указывать)
        "index_col": "Id",
        # колонка таргет
        "target_col": "SalePrice"
    },
    "training": {
        # стандартный lr
        "learning_rate": 0.1,
        # стандартная глубина дерева для бустингов
        "boosting_tree_depth": 6,
    },
    "split": {
        # true - включена кросс-валидация вместо одиночного сплита
        "cv": True,
        # true - перемешивать трейн и тест датасеты
        "shuffle": True,
        # true - стратифицировать при сплите
        "stratify": False,
        # если cv=False, в разбитие на трейн и тест это доля теста
        "test_size": 0.2, 
        # если cv=True, это количество фолдов
        "n_folds": 5
    },
    "preprocessing": {
        # дефолтные параметры для каждого препроцессора, передаются в конструктор
        "registry": {
            "base_col_dropper": {
                "columns": []
            },
            "base_mean_imputer": {
                "columns": [
                    "LotFrontage", 
                    "GarageYrBlt", 
                    "MasVnrArea", 
                    "BsmtFullBath", 
                    "BsmtHalfBath",
                    "BsmtFinSF2",
                    "TotalBsmtSF",
                    "BsmtUnfSF",
                    "BsmtFinSF1",
                    "GarageCars",
                    "GarageArea"
                ]
            },
            "base_onehot": {
                "columns": cat_features
            },
            "base_scaler": {
                "standard_cols": [
                    "MSSubClass",
                    "LotFrontage",
                    "LotArea",
                    "OverallQual",
                    "OverallCond",
                    "YearBuilt",
                    "YearRemodAdd",
                    "MasVnrArea",
                    "BsmtFinSF1",
                    "BsmtFinSF2",
                    "BsmtUnfSF",
                    "TotalBsmtSF",
                    "1stFlrSF",
                    "2ndFlrSF",
                    "LowQualFinSF",
                    "GrLivArea",
                    "BsmtFullBath",
                    "BsmtHalfBath",
                    "FullBath",
                    "HalfBath",
                    "BedroomAbvGr",
                    "KitchenAbvGr",
                    "TotRmsAbvGrd",
                    "Fireplaces",
                    "GarageYrBlt",
                    "GarageCars",
                    "GarageArea",
                    "WoodDeckSF",
                    "OpenPorchSF",
                    "EnclosedPorch",
                    "3SsnPorch",
                    "ScreenPorch",
                    "PoolArea",
                    "MiscVal",
                    "MoSold",
                    "YrSold",
                ],
                "minmax_cols": []
            },
            "log_transformer": {
                "columns": ["MSSubClass", "LotFrontage", "LotArea"]
            }
        },
        # дефолтный список препроцессоров, применяется, если в конфиге модели preprocessing: default
        "default": [
            # {
            #     "name": "feature_adder"
            #     # тут еще можно указать params, они переопределят параметры из preprocesing.registry
            #     # "params": {}
            # },
            {
                "name": "base_mean_imputer"
            },
            {
                "name": "log_transformer"
            },
            {
                "name": "base_onehot"
            },
            {
                "name": "base_scaler"
            }
        ]
    },
    # дефолтные гиперпараметры для каждой модели
    "models": {
        # название должно быть таким же, какое оно в реестре моделей (например в CLASSIFICATION_REGISTRY)
        "dummy": {
            # значения: default/custom
            # custom - список препроцессоров берется из preprocessing_steps
            "preprocessing": "custom",
            # для dummy не нужны препроцессоры, список должен быть в том же формате, что и preprocessing.default
            "preprocessing_steps": [],
            # все поля внутри params передаются в конструктор класса sklearn
            # но тут указаны дефолтные гиперпараметры и их можно переопределить в experiment.to_train
            "params": {
                "strategy": "mean"
            }
        },
        "linear": {
            # дефолтный список препроцессоров из preprocessing.default
            "preprocessing": "default",
            # дефолтные гиперпараметры выбраны перебором
            # это гиперпараметры, которые показывают лучшие метрики
            "params": {
                "alpha": 9.0,
                "l1_ratio": 0,
                "max_iter": 500
            },
        },
        "ridge": {
            "preprocessing": "default",
            "params": {
                "alpha": 1.0,
            },
        },
        "lasso": {
            "preprocessing": "default",
            "params": {
                "alpha": 1.0,
            },
        },
        "knn": {
            "preprocessing": "default",
            "params": {
                "n_neighbors": 9
            },
        },
        "decision_tree": {
            "preprocessing": "default",
            "params": {
                "random_state": "${general.seed}",
                "min_samples_leaf": 5
            },
        },
        "random_forest": {
            "preprocessing": "default",
            "params": {
                "n_estimators": 300,
                "max_depth": 12,
                "random_state": "${general.seed}"
            },
        },
        "catboost": {
            "preprocessing": "custom",
            # убрать из препроцессинга one hot (для catboost не рекомендуется)
            "preprocessing_steps": [
                {
                    "name": "base_mean_imputer"
                },
                {
                    "name": "log_transformer"
                },
                {
                    "name": "base_scaler"
                }
            ],
            "params": {
                "iterations": 300,
                "depth": "${training.boosting_tree_depth}",
                "cat_features": cat_features,
                "learning_rate": "${training.learning_rate}",
                "random_state": "${general.seed}",
                "verbose": 0
            }
        },
        "lightgbm": {
            "preprocessing": "default",
            "params": {
                "n_estimators": 300,
                "max_depth": "${training.boosting_tree_depth}",
                "learning_rate": "${training.learning_rate}",
                "random_state": "${general.seed}",
                "verbose": -1
            }
        },
        "xgboost": {
            "preprocessing": "default",
            "params": {
                "n_estimators": 300,
                "max_depth": "${training.boosting_tree_depth}",
                "eval_metric": "logloss",
                "learning_rate": "${training.learning_rate}",
                "random_state": "${general.seed}"
            }
        },
        "dnn": {
            "preprocessing": "default",
            # true - передавать в объект модели данные валидации перед fit
            # это такой костыль для адаптации DNN в флоу пайплайна, потому что сплит на трейн и вал происходит вне объекта модели
            "set_val_data_before_fit": True,
            # эти гиперпараметры передаются в конструктор адаптера DNN
            "params": {
                # размер входного слоя
                "in_features": 12,
                # скрытые слои, каждый элемент - размер слоя
                "hidden_sizes": [128, 64],
                # размер выходного слоя
                "out_features": 2,
                # вероятность дропаута
                "dropout_rate": 0.25,
                # размер батча
                "batch_size": 16,
                # размер тестового сплита
                "test_size": "${split.test_size}",
                # кол-во эпох (может быть и меньше, если early stopping)
                "epochs": 100,
                # скорость обучения
                "learning_rate": "${training.learning_rate}",
                # кол-во эпох без улучшения для early stopping
                "epochs_patience": 10,
                # порог значимого улучшения лосса для сброса patience
                "best_loss_threshold_to_save": 0.001,
                # сид
                "random_state": "${general.seed}",
                # директория для сохранения чекпоинтов
                "save_dir": "${paths.pytorch_models_dir}"
            }
        }
    },
    "experiment": {
        # метрика, должна быть из списка метрик, которые поддерживаются sklearn.metrics.get_scorer
        "metric": "neg_root_mean_squared_error",
        # список независимых шагов экспериментов
        "to_train": [
            {
                "model": "dummy"
            },
            {
                # название модели в models
                "model": "linear",
                # если указаны гиперпараметры, то они перезаписывают гиперпараметры из models
                "params": {
                    "alpha": 1.0
                }
            },
            {
                "model": "knn",
                "params": {
                    "n_neighbors": 3
                }
            },
            {
                "model": "decision_tree"
            },
            {
                "model": "random_forest"
            },
            {
                "model": "catboost"
            },
            {
                "model": "lightgbm"
            },
            {
                "model": "xgboost"
            },
            # {
            #     "model": "dnn",
            #     # переопределить флаг кросс-валидации для конкретного шага
            #     "cv": False,
            #     "params": {
            #         # переопределить lr из training.learning_rate
            #         "learning_rate": 0.001
            #     }
            # }
        ],
        "prediction": {
            # each - каждый эксперимент
            # best - лучший по метрике эксперимент
            "strategy": "best",
            # vote - голосование фолдов
            # best - лучший по метрике фолд
            "fold_strategy": "best",
            # если fold_strategy = vote, это
            # порог положительной классификации для голосования
            # "fold_vote_threshold": 0.5
        }
    },
}

regression_config = OmegaConf.create(regression_config)

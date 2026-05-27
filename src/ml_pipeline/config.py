from omegaconf import OmegaConf

# fmt: off
config = {
    "general": {
        "experiment_name": "lin&tree", 
        "seed": 42,
        "verbose": False
    },
    "paths": {
        "train": "data/train.csv", 
        "test": "data/test.csv",
        "submissions_dir": "submissions",
        "pytorch_models_dir": "pytorch_models"
    },
    "data": {
        "index_col": "PassengerId",
        "target_col": "Survived"
    },
    "training": {
        "learning_rate": 0.1,
        "boosting_tree_depth": 6,
    },
    "split": {
        "cv": True,
        "shuffle": True,
        # если cv=False, в разбитие на трейн и тест это доля теста
        "test_size": 0.2, 
        # если cv=True, это количество фолдов
        "n_folds": 5
    },
    "preprocessing": {
        "registry": {
            "feature_adder": {},
            "feature_dropper": {
                # дропнуть name, ticket, cabin, initial - не можем вытащить инфу из этих фичей
                # дропнуть embarked - будем использовать one-hot
                # дропнуть age и fare потому что будем использовать bin-ы вместо непрерывных фич
                "columns": ["Name", "Ticket", "Cabin", "Initial", "Embarked", "Age", "Fare"]
            },
            "imputer": {},
            "cat_encoder": {},
            "cont_encoder": {},
            "scaler": {
                "num_cols": [],
                "mm_cols": ["Ticket_Type"]
            }
        },
        "default": [
            {
                "name": "feature_adder"
            },
            {
                "name": "imputer"
            },
            {
                "name": "cat_encoder"
            },
            {
                "name": "cont_encoder"
            },
            {
                "name": "feature_dropper"
            },
            {
                "name": "scaler"
            }
        ]
    },
    "models": {
        "dummy": {
            "preprocessing": "custom",
            "preprocessing_steps": [],
            "params": {
                "strategy": "most_frequent"
            }
        },
        "logistic_regression": {
            "preprocessing": "default",
            "params": {
                "l1_ratio": 0, 
                "solver": "lbfgs",
                "max_iter": 500
            },
        },
        "knn": {
            "preprocessing": "default",
            "params": {
                "n_neighbors": 7
            },
        },
        "decision_tree": {
            "preprocessing": "default",
            "params": {
                "random_state": "${general.seed}",
                "min_samples_leaf": 4
            },
        },
        "random_forest": {
            "preprocessing": "default",
            "params": {
                "n_estimators": 700,
                "max_depth": 3,
                "random_state": "${general.seed}"
            },
        },
        "catboost": {
            "preprocessing": "default",
            "params": {
                "iterations": 300,
                "depth": "${training.boosting_tree_depth}",
                "cat_features": ["Sex", "Pclass", "SibSp", "Parch", "Alone", "Age_Group", "Fare_Range"],
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
            "params": {
                # nn layers
                "in_features": 12,
                "hidden_sizes": [128, 64],
                "out_features": 2,
                # nn params
                "dropout_rate": 0.25,
                # data split params
                "batch_size": 16,
                "test_size": "${split.test_size}",
                # training params
                "epochs": 100,
                "learning_rate": "${training.learning_rate}",
                "epochs_patience": 10,
                "best_loss_threshold_to_save": 0.001,
                # general params
                "random_state": "${general.seed}",
                "save_dir": "${paths.pytorch_models_dir}"
            }
        }
    },
    "experiment": {
        "metric": "accuracy",
        "to_train": [
            {
                "model": "dummy"
            },
            {
                "model": "logistic_regression",
                "params": {
                    "l1_ratio": 0
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
            {
                "model": "dnn",
                "params": {
                    "learning_rate": 0.001
                }
            }
        ],
        "prediction": {
            # each - каждый эксперимент
            # best - лучший по метрике эксперимент
            "strategy": "best",
            # vote - голосование фолдов
            # best - лучший по метрике фолд
            "fold_strategy": "best",
            # порог положительной классификации для голосования
            "fold_vote_threshold": 0.5
        }
    },
}

config = OmegaConf.create(config)

from omegaconf import OmegaConf

# fmt: off
config = {
    "general": {
        "experiment_name": "lin&tree", 
        "seed": 42
    },
    "paths": {
        "train": "data/train.csv", 
        "test": "data/test.csv",
        "submissions_dir": "submissions"
    },
    "data": {
        "target_col": "Survived"
    },
    "training": {
        "lr": 1e-4
    },
    "split": {
        "shuffle": True, 
        "test_size": 0.2, 
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
            "cont_encoder": {}
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
            }
        ],
        "prediction": {
            # each / best
            "strategy": "each"
        }
    },
}

config = OmegaConf.create(config)

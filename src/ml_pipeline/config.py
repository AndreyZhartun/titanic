from omegaconf import OmegaConf

# fmt: off
config = {
    "general": {
        "experiment_name": "baseline", 
        "seed": 42
    },
    "paths": {
        "train": "data/train.csv", 
        "test": "data/test.csv"
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
        "feature_adder": {},
        "feature_dropper": {},
        "imputer": {},
        "cat_encoder": {},
        "cont_encoder": {}
    },
    "models": {
        "dummy": {},
        "logistic_regression": {
            "preprocessing_steps": [
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
            ],
            "params": {
                "l1_ratio": 0, 
                "solver": "lbfgs"
            },
        },
        "knn": {
            "preprocessing_steps": [
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
            ],
            "params": {
                "n_neighbors": 3
            },
        },
        "decision_tree": {
            "preprocessing_steps": [
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
            ],
            "params": {
                "random_state": "${general.seed}"
            },
        },
        "random_forest": {
            "preprocessing_steps": [
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
            ],
            "params": {
                "n_estimators": 100,
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
            }
        ]
    },
}

config = OmegaConf.create(config)

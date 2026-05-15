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
        "cv_n_folds": 5
    },
    "preprocessing": {
        "feature_adder": {},
        "feature_dropper": {},
        "imputer": {},
        "cat_encoder": {},
        "cont_encoder": {}
    },
    "models": {
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
        }
    },
    "experiment": {
        "metric": "accuracy",
        "to_train": [
            {
                "model": "logistic_regression",
                "params": {
                    "l1_ratio": 0
                }
            }
        ]
    },
}

config = OmegaConf.create(config)

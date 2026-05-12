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
    "training": {
        "lr": 1e-4
    },
    "split": {
        "shuffle": True, 
        "test_size": 0.2, 
        "cv_n_folds": 5
    },
    "models": {
        "logistic_regression": {
            "preprocessing_steps": [
                
            ],
            "params": {
                "l1_ratio": 0, 
                "solver": "lbfgs"
            },
        }
    },
}

config = OmegaConf.create(config)

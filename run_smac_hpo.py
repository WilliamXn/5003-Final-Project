import numpy as np
import logging
import os
import json
import pandas as pd
from src.hpo.smac3_hpo import SMAC3HyperbandOptimizer
from src.models.lgb_model import LightGBMModel
from src.models.mlp_model import MLPModel
from src.models.svm_model import SVMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):  # Fix for numpy.bool_
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def get_objective(model_class, X, y, n_folds=5):
    """
    Create objective function for SMAC
    """
    def objective(params):
        # Create model with params
        model = model_class(params=params, n_folds=n_folds)
        # Train with CV and return score (logloss)
        score = model.train_with_cv(X, y, verbose=False)
        return score
    return objective

def run_hpo(model_name, X_train, y_train, n_trials=50, cv_folds=5):
    """
    Run HPO for a specific model
    """
    logger.info(f"Starting HPO for {model_name}...")
    
    # Select model class
    if model_name == 'lightgbm':
        model_class = LightGBMModel
    elif model_name == 'mlp':
        model_class = MLPModel
    elif model_name == 'svm':
        model_class = SVMModel
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create optimizer
    optimizer = SMAC3HyperbandOptimizer(
        model_name=model_name,
        n_trials=n_trials,
        cv_folds=cv_folds
    )
    
    # Create objective function
    objective = get_objective(model_class, X_train, y_train, n_folds=cv_folds)
    
    # Run optimization
    best_params, best_score = optimizer.optimize(objective)
    
    logger.info(f"Best params for {model_name}: {best_params}")
    logger.info(f"Best score for {model_name}: {best_score}")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open(f'outputs/smac3_{model_name}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4, cls=NpEncoder)
        
    return best_params, best_score

def main():
    # Load data
    logger.info("Loading data...")
    try:
        X_train = np.load('data/processed/train_features.npy')
        y_train = np.load('data/processed/train_labels.npy')
    except FileNotFoundError:
        logger.error("Data files not found. Please run preprocessing first.")
        return

    # Run HPO for MLP only
    model_name = 'mlp'
    try:
        best_params, best_score = run_hpo(model_name, X_train, y_train, n_trials=50)
        results = {
            'best_params': best_params,
            'best_score': best_score
        }
        # Save results
        with open('outputs/smac3_mlp_results.json', 'w') as f:
            json.dump(results, f, indent=4, cls=NpEncoder)
        logger.info("HPO completed for MLP.")
    except Exception as e:
        logger.error(f"Error optimizing {model_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

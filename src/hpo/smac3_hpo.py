import logging
import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from .base_hpo import BaseHPO

logger = logging.getLogger(__name__)

class SMAC3HyperbandOptimizer(BaseHPO):
    """
    SMAC3 Hyperband Optimizer
    """
    
    def __init__(self, model_name='lightgbm', n_trials=50, cv_folds=5, random_state=42, 
                 min_budget=None, max_budget=None):
        """
        Initialize SMAC3 Hyperband Optimizer
        
        Args:
            model_name: Model name ('lightgbm', 'svm', 'mlp')
            n_trials: Number of trials (ignored by Hyperband usually, but used for scenario)
            cv_folds: Number of CV folds
            random_state: Random seed
            min_budget: Minimum budget (e.g., n_estimators or max_iter)
            max_budget: Maximum budget
        """
        super().__init__(model_name, n_trials, cv_folds, random_state)
        
        # Set default budgets if not provided
        if min_budget is None or max_budget is None:
            if model_name == 'lightgbm':
                self.min_budget = 20
                self.max_budget = 500  # Reduced from 1000 to save time, or keep 1000
            elif model_name == 'mlp':
                self.min_budget = 10
                self.max_budget = 200
            elif model_name == 'svm':
                self.min_budget = 100
                self.max_budget = 2000 # SVM converges slowly
            else:
                self.min_budget = 10
                self.max_budget = 100
        else:
            self.min_budget = min_budget
            self.max_budget = max_budget

    def _build_config_space(self):
        """
        Convert JSON search space to ConfigSpace
        """
        cs = ConfigurationSpace(seed=self.random_state)
        
        for param_name, param_spec in self.search_space.items():
            # Skip parameters that are used as fidelity
            if self.model_name == 'lightgbm' and param_name == 'n_estimators':
                continue
            if self.model_name == 'mlp' and param_name == 'max_iter':
                continue
            if self.model_name == 'svm' and param_name == 'max_iter':
                continue
            
            param_type = param_spec['type']
            
            if param_type == 'int':
                cs.add(Integer(
                    param_name, 
                    bounds=(param_spec['low'], param_spec['high']),
                    log=param_spec.get('log', False)
                ))
            elif param_type == 'float':
                cs.add(Float(
                    param_name, 
                    bounds=(param_spec['low'], param_spec['high']),
                    log=param_spec.get('log', False)
                ))
            elif param_type == 'categorical':
                # Handle list of lists (e.g. for MLP hidden_layer_sizes)
                choices = param_spec['choices']
                if choices and isinstance(choices[0], list):
                    # Convert lists to string representation for ConfigSpace
                    choices = [str(c) for c in choices]
                
                # Handle null/None
                choices = [c if c is not None else "None" for c in choices]
                
                cs.add(Categorical(param_name, choices))
                
        return cs

    def optimize(self, objective_function, verbose=True):
        """
        Execute SMAC3 Hyperband optimization
        """
        cs = self._build_config_space()
        
        # Define the target function for SMAC
        def train(config, seed: int = 0, budget: float = 100):
            params = dict(config)
            
            # Convert string representation back to list for MLP
            if self.model_name == 'mlp' and 'hidden_layer_sizes' in params:
                import ast
                try:
                    if isinstance(params['hidden_layer_sizes'], str):
                        params['hidden_layer_sizes'] = ast.literal_eval(params['hidden_layer_sizes'])
                except:
                    pass
            
            # Map budget to model specific parameter
            if self.model_name == 'lightgbm':
                params['n_estimators'] = int(budget)
            elif self.model_name == 'mlp':
                params['max_iter'] = int(budget)
            elif self.model_name == 'svm':
                params['max_iter'] = int(budget)
            
            # Handle "None" string for all parameters
            for k, v in params.items():
                if v == "None":
                    params[k] = None
            
            # Call the objective function provided by the user
            return objective_function(params)

        # Scenario definition
        scenario = Scenario(
            cs,
            deterministic=True, # CV is deterministic with fixed seed
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            n_trials=self.n_trials,
            objectives="cost", # Minimize cost (logloss)
            seed=self.random_state
        )

        # Create SMAC object with MultiFidelityFacade (Hyperband)
        smac = MFFacade(
            scenario,
            train,
            overwrite=True, # Overwrite previous run results
            logging_level=logging.INFO if verbose else logging.WARNING,
        )

        # Optimize
        incumbent = smac.optimize()
        
        # Get best config and cost (validate on max budget)
        best_cost = train(incumbent, seed=self.random_state, budget=self.max_budget)
        
        # Convert to dict
        best_params = dict(incumbent)
        
        # Post-processing for MLP hidden_layer_sizes
        if self.model_name == 'mlp' and 'hidden_layer_sizes' in best_params:
             import ast
             try:
                 if isinstance(best_params['hidden_layer_sizes'], str):
                     best_params['hidden_layer_sizes'] = ast.literal_eval(best_params['hidden_layer_sizes'])
             except:
                 pass
                 
        # Add the budget parameter to best_params (max budget)
        if self.model_name == 'lightgbm':
            best_params['n_estimators'] = int(self.max_budget)
        elif self.model_name == 'mlp':
            best_params['max_iter'] = int(self.max_budget)
        elif self.model_name == 'svm':
            best_params['max_iter'] = int(self.max_budget)

        # Save to history
        self.history.append({
            'params': best_params,
            'score': best_cost
        })

        return best_params, best_cost


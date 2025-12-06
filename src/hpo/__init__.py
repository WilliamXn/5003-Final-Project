from .base_hpo import BaseHPO
from .random_search import RandomSearch
from .grid_search import GridSearch
from .tpe_optuna import TPEOptuna
from .smac_optimizer import SMACOptimizer
from .smac3_hpo import SMAC3HyperbandOptimizer

# HPO算法注册表
# 队友完成各自的算法后，在这里注册即可自动生效
AVAILABLE_ALGORITHMS = {
    'random': RandomSearch,      # A同学 - 已完成
    'grid': GridSearch,          # B同学 - 待完成
    'tpe': TPEOptuna,            # C同学 - 待完成
    'smac': SMACOptimizer,       # D同学 - 待完成 (OpenBox)
    'smac3': SMAC3HyperbandOptimizer, # SMAC3 Hyperband
}


def get_hpo_algorithm(algorithm_name):
    """
    获取HPO算法类
    
    Args:
        algorithm_name: 算法名称（如 'random', 'grid', 'tpe', 'smac'）
        
    Returns:
        HPO算法类
        
    Raises:
        ValueError: 如果算法名称不存在
    """
    if algorithm_name not in AVAILABLE_ALGORITHMS:
        available = ', '.join(AVAILABLE_ALGORITHMS.keys())
        raise ValueError(
            f"未知的HPO算法: '{algorithm_name}'\n"
            f"可用的算法: {available}"
        )
    return AVAILABLE_ALGORITHMS[algorithm_name]


def list_algorithms():
    """列出所有可用的HPO算法"""
    return list(AVAILABLE_ALGORITHMS.keys())


__all__ = [
    'BaseHPO', 
    'RandomSearch', 
    'GridSearch',
    'TPEOptuna',
    'SMACOptimizer',
    'SMAC3HyperbandOptimizer',
    'get_hpo_algorithm',  
    'list_algorithms', 
    'AVAILABLE_ALGORITHMS'
]

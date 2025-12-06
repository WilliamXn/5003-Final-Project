import numpy as np
import pandas as pd
import os
import json
import logging
from src.models.mlp_model import MLPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载数据
logger.info("Loading data...")
train_features = np.load("data/processed/train_features.npy")
train_labels = np.load("data/processed/train_labels.npy")
test_features = np.load("data/processed/test_features.npy")
test_ids = np.load("data/processed/test_ids.npy")

# 加载最佳参数
logger.info("Loading best parameters...")
with open("outputs/smac3_mlp_best_params.json", "r") as f:
    best_params = json.load(f)

logger.info(f"Best parameters: {best_params}")

# 初始化 MLP 模型
logger.info("Initializing MLP model...")
model = MLPModel(params=best_params, n_folds=5)

# 使用最佳参数训练模型
logger.info("Training MLP model with best parameters...")
model.train_with_cv(train_features, train_labels, verbose=True)

# 对测试集进行预测（返回概率）
logger.info("Generating predictions for test set...")
test_predictions = model.predict(test_features)

logger.info(f"Predictions shape: {test_predictions.shape}")
logger.info(f"First 5 predictions:\n{test_predictions[:5]}")

# 确保有 7 列（7 个类别）
if test_predictions.shape[1] != 7:
    logger.warning(f"Expected 7 columns, but got {test_predictions.shape[1]}")

# 生成提交文件
logger.info("Creating submission DataFrame...")
submission = pd.DataFrame({
    'id': test_ids.astype(int),
    'target_0': test_predictions[:, 0],
    'target_1': test_predictions[:, 1],
    'target_2': test_predictions[:, 2],
    'target_3': test_predictions[:, 3],
    'target_4': test_predictions[:, 4],
    'target_5': test_predictions[:, 5],
    'target_6': test_predictions[:, 6]
})

logger.info(f"Submission DataFrame:\n{submission.head()}")

# 保存提交文件
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "mlp_submission.csv")
submission.to_csv(output_path, index=False)

logger.info(f"Submission file saved to: {output_path}")
print(f"提交文件已保存: {output_path}")
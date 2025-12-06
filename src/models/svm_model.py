import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from .base_model import BaseModel

# 抑制SVM收敛警告（数据已标准化，警告不影响结果）
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class SVMModel(BaseModel):
    """SVM模型（带概率校准）"""
    
    def _create_model(self, params):
        """创建SVM模型"""
        # 从params中提取参数
        svm_params = {
            'C': params.get('C', 1.0),
            'kernel': params.get('kernel', 'rbf'),
            'gamma': params.get('gamma', 'scale'),
            'probability': True,  # 必须开启以支持概率预测
            'random_state': self.random_state,
            'max_iter': params.get('max_iter', 10000),  # 增加迭代次数，配合StandardScaler使用
        }
        
        # 如果是poly核，添加degree参数
        if svm_params['kernel'] == 'poly':
            svm_params['degree'] = params.get('degree', 3)
        
        # class_weight参数
        if 'class_weight' in params and params['class_weight'] is not None:
            svm_params['class_weight'] = params['class_weight']
        
        # 创建基础SVM
        svm = SVC(**svm_params)
        
        # 使用CalibratedClassifierCV进行概率校准（提高概率质量）
        # 注意：这会增加训练时间，但提高logloss性能
        model = CalibratedClassifierCV(svm, cv=3, method='sigmoid')
        
        return model
    
    def _fit(self, model, X_train, y_train, X_val=None, y_val=None):
        """训练SVM模型"""
        # SVM不支持early stopping，直接训练
        model.fit(X_train, y_train)
        return model
    
    def _predict_proba(self, model, X):
        """预测概率"""
        return model.predict_proba(X)


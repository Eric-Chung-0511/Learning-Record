import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer


# 定義參數搜索空間
search_spaces = {
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
    'max_depth': Integer(3, 10),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.5, 1.0, 'uniform'),
    'colsample_bytree': Real(0.5, 1.0, 'uniform'),
    'n_estimators': Integer(100, 1000)
}

# 創建XGBoost分類器
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 設置BayesSearchCV
bayes_search = BayesSearchCV(
    xgb,
    search_spaces,
    n_iter=50,  # 總迭代次數
    cv=3,  # 交叉驗證折數
    n_jobs=-1,  # 使用所有可用的CPU核心
    verbose=1,
    random_state=42,
    scoring='accuracy'
)

# 執行搜索
bayes_search.fit(X_train, y_train)

# 輸出最佳參數和分數
print("最佳參數:", bayes_search.best_params_)
print("最佳交叉驗證分數:", bayes_search.best_score_)

# 用最佳參數在測試集上評估模型
best_model = bayes_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("測試集準確率:", accuracy)

from sklearn import model_selection 
import numpy as np
import pandas as pd
import xgboost as xgb

bioresponce = pd.read_csv('bioresponse.csv', header=0, sep=',')
bioresponce_target = bioresponce.Activity.values
bioresponce_data = bioresponce.iloc[:, 1:]
n_trees = [1] + list(range(10, 55, 5) )
scoring_xgb = []
for n_tree in n_trees:
    estimator = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=n_tree, min_child_weight=3)
    score = model_selection.cross_val_score(estimator, bioresponce_data, bioresponce_target, 
                                             scoring = 'accuracy', cv = 3)    
    scoring_xgb.append(score)
print(scoring_xgb)
scoring_xgb = np.asmatrix(scoring_xgb)
"""
Редактор Spyder

Это временный скриптовый файл.
"""


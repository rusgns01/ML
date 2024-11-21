import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv("https://bit.ly/wine_csv_data")
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# n_jobs = -1 -< 모든 cpu코어 활용
rf = RandomForestClassifier(n_jobs = -1, random_state = 42)

# 병렬로 교차검증
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs = -1)
# 교차 검증 정확도
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 모델을 훈련 세트에 맞게 학습 후 특성 중요도 프린트
rf.fit(train_input, train_target)
print(rf.feature_importances_)

rf = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 42)
rf.fit(train_input, train_target)
print(rf.oob_score_)

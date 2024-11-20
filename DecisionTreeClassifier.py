import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# wine.describe()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

dt = DecisionTreeClassifier(max_depth=5, random_state = 42)
dt.fit(train_scaled, train_target)

# 최대 깊이 3
#dt = DecisionTreeClassifier(max_depth=3, random_state = 42)
#dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# 최대 깊이 제한 없음
#dt = DecisionTreeClassifier(random_state = 42)
#dt.fit(train_scaled, train_target)

#print(dt.score(train_scaled, train_target))
#print(dt.score(test_scaled, test_target))

# 깊이 제한 없는 트리 시각화
#plt.figure(figsize = (10, 7))
#plot_tree(dt)
#plt.show()

# 최대 깊이 1 시각화
#plt.figure(figsize = (10, 7))

#plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
#plt.show()
# sugar = 테스트 조건, gini = 불순도, samles = 총 샘플 수, value = 클래스별 샘플 수

# 특성 중요도
#print(dt.feature_importances_)

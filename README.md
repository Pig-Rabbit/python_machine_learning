

1. Loading and Review the Data
- Use the Pandas library for exploring and manipulating data
```python
import pandas as pd
file_path = '../desktop/private/train.csv'
data = pd.read_csv(data)
data.describe()
```

2. Selecting Data for Modeling
- Specify prediction target
- Create a list with the predictive features
```python
data.colums # find the name of the prediction target
y = data['target']
feature_names = ['feature1','feature2','feature3']
X = data[feature_names]
```

3. Specify and Fit Model
- Import DecisionTreeRegressor from sklearn
- [random_state] ensures the same results in each run
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 1)
model.fit(X,y)
```

4. Make Predictins
```python
predictions = model.predict(X)
print(predictions)
```

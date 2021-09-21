**Imported Modules**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

## Decision Tree
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
feature_names = ['feature1', 'feature2', 'feature3']
X = data[feature_names]
```

3. Split Data
```python
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
```

4. Specify and Fit Model
- Import DecisionTreeRegressor from sklearn
- [random_state] ensures the same results in each run
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 1)
model.fit(train_X, train_y)
```

5. Make Predictions
```python
val_predictions = model.predict(val_X)
print(val_predictions)
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_predictions, val_y)
print(val_mae)
```

6. Fine-tune model for better performance
- Underfitting and Overfitting
```python
def get_mae(max_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_lead_nodes = max_nodes, random_state = 0)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_predictions, val_y)
    return(mae)

candidate_max_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_nodes}
best_tree_size = min(scores, key = scores.get)
print(best_tree_size)
```

7. Fit Model Using All Data
- Make model more accuracy by using all of the data and keeping the best tree size
```python
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 1)
final_model.fit(X, y)
```

## Random Forest
1. Use a Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
rf_val_mae = mean_absolute_error(val_predictions, val_y)
```

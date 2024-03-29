# The score of this code is 0.77511

# load data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

# calculate rate of women and men survived
women = train_data.loc[train_data['Sex'] == 'female']['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived: ", rate_women)
men = train_data.loc[train_data['Sex'] == 'male']['Survived']
rate_men = sum(men)/len(men)
print("% of men who survived: ", rate_men)

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
feature = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
y = train_data["Survived"]

model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print(output.head())
print('submission was successfully saved')

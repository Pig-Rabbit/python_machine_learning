# Recommend Competition List
# 1. Titanic: Machine Learning from Disaster
# 2. House Prices: Advanced Regression Techniques
# 3. Digit Recognizer

import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble mport GradientBoostingClassfier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklear.svm import SVC

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.info() # info about data
# Percentage of NA per property sorted
p = (train_data.isna().sum()/len(train_data)*100).sort_values(ascending=False)
# Unique values for duplications and other useful info
u = train_data.nunique().sort_values()

# Data Cleaning
# 1. Missing values
# 1.1. >50%: drop
# 1.2. ~20%: try to fill
# 1.3. <0.5%: drop the cases
# 2. Categorical values
# 2.1. two values: use label encoder
# 2.2. various data such as name: drop

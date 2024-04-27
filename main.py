# %%
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# handling missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

# models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBModel

# handle outliers
from sklearn.ensemble import IsolationForest

# feature encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# model selection
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV

# hyper-parameter tuning 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from itertools import product

from util import plotter, cat_plotter

# %%
import warnings
warnings.filterwarnings('ignore') 

pd.options.display.max_columns = 40

# %%
"""
# Objectives
"""

# %%


# %%
"""
# Loading Data
"""

# %%
df = pd.read_csv("./kidney_disease.csv")
df.head()

# %%
# rename the columns
columns = ['id', 'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']
df.columns = columns
df.head()


# %%
"""
# Data Preprocessing
"""

# %%
"""
#### fixing some problmes in dataset  
"""

# %%
df = df.drop(columns=["id"])

# %%
"""
for this you could also use `replace` :
```python
    df[feature].replace(
        {
            "\t?": np.nan
        }
    )
```
"""

# %%
# one wired thing is that there is a value \t? in one of the columns, and we should fix it...
faulty_columns = ["packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count"]
for column in faulty_columns :
    index = df[column].where(df[column] == "\t?").dropna().index[0]
    df[column][index] = np.nan

df[faulty_columns] = df[faulty_columns].astype("float") # fixing their dtype

# %%
# the values in this column look like this which should be fiexed
# no       258
# yes      134
# \tno       3
# \tyes      2
#  yes       1
 
index = df["diabetes_mellitus"].where(df["diabetes_mellitus"] == "\tno").dropna().index
df["diabetes_mellitus"][index] = "no"

index = df["diabetes_mellitus"].where(df["diabetes_mellitus"] == "\tyes").dropna().index
df["diabetes_mellitus"][index] = "yes"

index = df["diabetes_mellitus"].where(df["diabetes_mellitus"] == " yes").dropna().index
df["diabetes_mellitus"][index] = "yes"

# %%
# coronary_artery_disease
# no      362
# yes      34
# \tno      2

index = df["coronary_artery_disease"].where(df["coronary_artery_disease"] == "\tno").dropna().index
df["coronary_artery_disease"][index] = "no"

# %%
# class
# ckd       248
# notckd    150
# ckd\t       2

index = df["class"].where(df["class"] == "ckd\t").dropna().index
df["class"][index] = "ckd"

# %%
"""
## fixing null values
"""

# %%
df.isnull().sum()

# %%
"""
#### examining some of the columns **Distribution** and their **BoxPlot**
"""

# %%
numeric_features = df.select_dtypes(["int", "float"]).columns.to_numpy()

for column in numeric_features :
    plotter(df, column)


# %%
"""
**after this noticed that `sugar` and `albumin` are *categorical* data**
"""

# %%
cat_features = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
       'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
       'appetite', 'peda_edema', 'aanemia', 'sugar', 'albumin', 'class']

fig_x = 3
fig_y = 5

fig, axes = plt.subplots(fig_x, fig_y, figsize=(10, 5), layout="constrained")

for index, column in zip(product(range(fig_x), range(fig_y)), cat_features) :
    cat_plotter(df, column, ax=axes[index[0], index[1]])

plt.show()


# %%
"""
#### Imputation
"""

# %%
# using Iterative Imputer
def IterativeImputerFunc(df, features, **kwargs) :
    imputer = IterativeImputer(**kwargs)
    imputed_feature = imputer.fit_transform(df.select_dtypes(["int", "float"]))
    return imputed_feature

# %%
# using KNN imputer
def KNNImputerFunc(df, feature: str, **kwargs) :
    KNN_imputer = KNNImputer(n_neighbors=5, **kwargs)
    KNN_imputed_feature = KNN_imputer.fit_transform(df[feature].to_numpy().reshape(-1, 1))
    return KNN_imputed_feature

# %%
"""
#### appling null value fixing methods
"""

# %%
features = [
    "blood_glucose_random", "blood_urea", "sodium", "potassium", "haemoglobin",
    "packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count", "serum_creatinine"
]

# %%
values = IterativeImputerFunc(df, features) # new values for columns that have Nan in them
tmp = df.select_dtypes(["int", "float"]).copy() # a tmp dataframe to store new values in it
tmp[:] = values
df = df.drop(columns=features) # dropping the features so that we can replace them later
df[features] = tmp[features]

# %%
"""
for some reasons they both set the `mean()` for all values ...
"""

# %%
# replacing these features with their mode
for feature in ["age"] :
    filler = df[feature].mode()[0]
    df[feature] = df[feature].fillna(filler)

# %%
# replacing categorical data's missing values with their top n values
features = [
    ("blood_pressure", 5),
    ("specific_gravity", 4),
    ("albumin", 5),
    ("sugar", 5),
    ("pus_cell", 2),
    ("pus_cell_clumps", 2),
    ("bacteria", 2),
    ("hypertension", 2),
    ('diabetes_mellitus', 2),
    ('coronary_artery_disease', 2),
    ('appetite', 2),
    ('peda_edema', 2),
    ('aanemia', 2),
]


for feature_name, group_count in features : 
    group_chance = df[feature_name].value_counts() / df[feature_name].value_counts().sum()
    group_chance = group_chance.sort_values(ascending=False).iloc[0:group_count]
    group_chance /= group_chance.sum() # normalize the data so that the values sum up to 1.0

    group_names = group_chance.index

    nan_index = df[feature_name][df[feature_name].isna()].index
    values = np.random.choice(
        group_names,
        size=len(nan_index),
        p=group_chance,
    )

    values = pd.Series( # convert the values to a series
        values,
        index=nan_index
    )

    df[feature_name] = df[feature_name].fillna(values)

# %%
# drop some of the features with many null values 
columns = ["red_blood_cells", ]
df = df.drop(columns=columns)

# %%
"""
The other way you can fill the Nan values is by getting **samples** from the non-Nan values in that column, and fill the Nan values with them
"""

# %%
df.isna().sum()

# %%
"""
## Fixing dtypes
"""

# %%
df.dtypes

# %%
# to get the columns that have only int values 
df.select_dtypes("float").apply(
    lambda column: column.apply(
            lambda value: value.is_integer(),
        ),
    axis=1
).all()

# %%
df.select_dtypes(exclude=["int", "float"]).value_counts()

# %%
# fixing dtypes
df.astype(
    {
        "age": int,
        "blood_pressure": int,
        "albumin": int,  # after checking the visualizations, i found that it is categorical data 
        "sugar": int,  # after checking the visualizations, i found that it is categorical data
        "pus_cell": "category",
        "pus_cell_clumps": "category",
        "bacteria": "category",
        "appetite": "category",
    }
)

# %%
"""
to fix the boolean features which are :
"""

# %%
"""
hypertension  
diabetes_mellitus  
coronary_artery_disease  
peda_edema  
aanemia  
class  
"""

# %%
# we can use label encoder :
cols_to_encode = ["hypertension", "diabetes_mellitus", "coronary_artery_disease", "peda_edema", "aanemia", 'class']
encoder = LabelEncoder()
for column in cols_to_encode :
    df[column] = encoder.fit_transform(df[column])


# %%
df.head()

# %%
"""
## EDA
"""

# %%
"""
- correlation
- violin plot
"""

# %%
"""
## Handling Outliers
"""

# %%
fig_x = 4
fig_y = 5
fig, axes = plt.subplots(fig_x, fig_y, figsize=(10, 5), layout="constrained")

for index, column in zip(product(range(fig_x), range(fig_y)), df.select_dtypes(["int", "float", "bool"])) :
    sns.boxplot(
        df[column],
        ax=axes[index[0], index[1]],
    )
    axes[index[0], index[1]].set_title(column)

# %%
fig_x = 4
fig_y = 5
fig, axes = plt.subplots(fig_x, fig_y, figsize=(10, 5), layout="constrained")

for index, column in zip(product(range(fig_x), range(fig_y)), df.select_dtypes(["int", "float", "bool"])) :
    sns.violinplot(
        df[column],
        ax=axes[index[0], index[1]],
    )
    axes[index[0], index[1]].set_title(column)

# %%
df["class"].value_counts()

# %%
"""
##### method 1:  `IsolationForest`
"""

# %%
def box_plotter(df, features=None) :
    fig_x = 3
    fig_y = 5
    fig, axes = plt.subplots(fig_x, fig_y, figsize=(10, 5), layout="constrained")
    df = df.select_dtypes(["int", "float"]) if features==None else df.select_dtypes(["int", "float"])[features]
    for index, column in zip(product(range(fig_x), range(fig_y)), df) :
        sns.boxplot(
            df[column],
            ax=axes[index[0], index[1]],
        )
        axes[index[0], index[1]].set_title(column)

# %%
def single_outlier_removal(column: pd.DataFrame, n_estimators=20, contamination=0.05) :
    # single feature
    isolation_forest =IsolationForest(n_estimators=n_estimators, max_samples='auto', contamination=contamination, max_features=1)
    isolation_forest.fit(column)

    tmp_df = column.copy()
    tmp_df["anomaly"] = isolation_forest.predict(column)
    tmp_df = tmp_df.query("anomaly == 1")
    tmp_df.drop(columns="anomaly", inplace=True)
    return tmp_df

# %%
def outlier_removal(df: pd.DataFrame, n_estimators=200, contamination=0.05) :
    # whole features
    isolation_forest =IsolationForest(n_estimators=n_estimators, max_samples='auto', contamination=contamination, max_features=1)
    isolation_forest.fit(df)

    tmp_df = df.copy()
    tmp_df["anomaly"] = isolation_forest.predict(df)
    tmp_df = tmp_df.query("anomaly == 1")
    tmp_df.drop(columns="anomaly", inplace=True)
    return tmp_df

# %%
new = outlier_removal(df.select_dtypes(["int", "float"]), n_estimators=200)
box_plotter(new)

# %%
fig_x = 3
fig_y = 5
fig, axes = plt.subplots(fig_x, fig_y, figsize=(10, 5), layout="constrained")

for index, column in zip(product(range(fig_x), range(fig_y)), df.select_dtypes(["int", "float"])) :
    sns.boxplot(
        single_outlier_removal(df[[column]]),
        ax=axes[index[0], index[1]],
    )
    axes[index[0], index[1]].set_title(column)

# %%
"""
one wired thing that i don't understand is that when i use `IsolationForest` for single features compared to all the features at the same time,  
the results are different , beside the fact that the `n_estimators` used in `single_outlier_removal()` is smaller than `outlier_removal()`
"""

# %%
"""
##### apply IsolationForest / `single_outlier_removal`
"""

# %%
chosen_indexes = set()
for feature in df.select_dtypes(["int", "float"]) :
    new = single_outlier_removal(df[[feature]])
    indexes = new.index.tolist()
    chosen_indexes = set(indexes) if not chosen_indexes else chosen_indexes
    chosen_indexes = chosen_indexes.intersection(set(indexes))

# %%
new_df = df.iloc[list(chosen_indexes), :]
new_df

# %%
box_plotter(new_df)

# %%
"""
### Handle imbalance data
"""

# %%


# %%
"""
## Feature Encoding
"""

# %%
"""
we have already handled some encodings in fixing-dtypes section, now let's do the rest ...
"""

# %%
df.head()

# %%
# we use label encoding
df[
    ["pus_cell", "pus_cell_clumps", "bacteria", "appetite"]
].value_counts()

# %%
"""
these features are also boolean, so we could also handle them in the previous parts
"""

# %%
encoder = LabelEncoder()
for column in ["pus_cell", "pus_cell_clumps", "bacteria", "appetite"] :
    df[column] = encoder.fit_transform(df[column])

# %%
df.head()

# %%
"""
## Feature Scaling
"""

# %%
# Standard scaler
scaler = StandardScaler()
new_data = scaler.fit_transform(df.drop(columns="class"))
scaled_df = pd.DataFrame(
    new_data,
    columns=df.drop(columns="class").columns
)
scaled_df = pd.concat([scaled_df, df["class"]], axis=1)

# %%
"""
## UseFull features
"""

# %%
corr = df.drop(columns="class").corrwith(df["class"]).to_frame(name="correlation")

fig, ax = plt.subplots(figsize=(15, 7))
ax = sns.heatmap(
    corr.sort_values(by="correlation"),
    annot=True, 
    cbar=False,
    cmap="coolwarm",
    center=0,
    ax=ax
)
ax.set_title("df correlation");

# %%
# without scaling
X = df.drop(columns="class")
y = df["class"]

l1_regularizer = Lasso(alpha=0.05)
selector = SelectFromModel(l1_regularizer)
selector.fit(X, y)

lasso_selected_features = X.columns[selector.get_support()]
lasso_selected_features


# %%
# another method using KBest
# for this method, the values should be non zero, so first we scale them using MinMax scaler
scaler = MinMaxScaler()
new_data = scaler.fit_transform(df.drop(columns="class"))
MinMax_scaled_df = pd.DataFrame(
    new_data,
    columns=df.drop(columns="class").columns
)
MinMax_scaled_df = pd.concat([MinMax_scaled_df, df["class"]], axis=1)

X = MinMax_scaled_df.drop(columns="class")
y = MinMax_scaled_df["class"]

selector = SelectKBest(chi2, k=6)
selector.fit(X, y)
KBest_features = X.columns[selector.get_support()]

# %%
"""
# Modeling
"""

# %%
"""
#### Logistic Regression
"""

# %%
# simple logistic regression 
# the features are selected based on the correlation
X = df.drop(columns="class")
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

selected_features = corr.query("correlation >= 0.5 | correlation <= -0.5").index
lr = LogisticRegression()
lr.fit(X_train[selected_features], y_train)
lr.score(X_test[selected_features], y_test)

# %%
# let's try it with the lasso selected features 
X = df.drop(columns="class")
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

lr = LogisticRegression()
lr.fit(X_train[lasso_selected_features], y_train)
lr.score(X_test[lasso_selected_features], y_test)

# %%
# let's try it with the KBest selected features 
X = df.drop(columns="class")
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

lr = LogisticRegression()
lr.fit(X_train[KBest_features], y_train)
lr.score(X_test[KBest_features], y_test)

# %%
"""
<br> <br>
"""

# %%
# now let's use the MinMax version of data :
X = MinMax_scaled_df.drop(columns="class")
y = MinMax_scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

selected_features = corr.query("correlation >= 0.5 | correlation <= -0.5").index
lr = LogisticRegression()
lr.fit(X_train[selected_features], y_train)
lr.score(X_test[selected_features], y_test)


# wow !!!

# %%
# now let's use the Standard scaled version of data :
X = scaled_df.drop(columns="class")
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

selected_features = corr.query("correlation >= 0.5 | correlation <= -0.5").index
lr = LogisticRegression()
lr.fit(X_train[selected_features], y_train)
lr.score(X_test[selected_features], y_test)


# wow !!!

# %%
"""
**scaling had an amazing effect !**
"""

# %%
"""
so as we see, we have 99 percent accuracy with Scaling and selecting the best features by Correlation(and without using  
the selected features by libraries).
"""

# %%
"""
#### Decision Tree
"""

# %%
X = scaled_df[selected_features] # `class` feature is not here
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt.score(X_test, y_test)

# %%
"""
#### Naive Bayes
"""

# %%
X = scaled_df[selected_features] # `class` feature is not here
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

nbc = GaussianNB()
nbc.fit(X_train, y_train)
nbc.score(X_test, y_test)

# %%
"""
#### Random Forest
"""

# %%
X = scaled_df[selected_features] # `class` feature is not here
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)

# %%
"""
#### XGBoost
"""

# %%
X = scaled_df[selected_features] # `class` feature is not here
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

params = {
    'max_depth': 5,
    'eta': 1,
    # 'objective': 'binary:logistic',
}

model = XGBModel()
model.fit(X_train, y_train, )
y_pred = model.predict(X_test)

# accuracy_score(y_test, y_pred)


# %%
"""
for some reasons the output is not a binary output, so i have to do it myself...
"""

# %%
accuracy_score(
    y_test,
    (y_pred > 0.5).astype(int),
)

# %%
"""
## HyperParameter tuning
"""

# %%
"""
#### HyperParameter tuning for RandomForest
"""

# %%
"""
##### RandomizedSearchCV
"""

# %%
# using RandomizedSearchCV
param_grid = {
    "n_estimators": np.arange(start=20, stop=500, step=50),
    "criterion": ['gini', 'entropy', 'log_loss'] ,
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 3, 5, 7],
}
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_grid,
    cv=5,
    scoring='accuracy',
    random_state=1502,
    verbose=5,
)

random_search.fit(X_train, y_train)

# %%
random_search.best_params_

# %%
random_search.best_score_

# %%
# checking all the checked situations which is really useful
pd.DataFrame(random_search.cv_results_).sort_values(by="rank_test_score")

# %%
"""
##### GirdSearchCV
"""

# %%
# using GirdSearchCV
# based on the previous random parameter checking, we have a sense of the range of parameters

param_grid = {
    "n_estimators": [70, 120, 300, 400],
    "criterion": ['log_loss'] ,
    "min_samples_split": [4, 6, 8, 10, 12],
    "min_samples_leaf": [1, 5, 7],

}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=5,
)

grid_search.fit(X_train, y_train)

# %%
grid_search.best_params_

# %%
grid_search.best_score_

# %%
pd.DataFrame(grid_search.cv_results_).sort_values(by=["rank_test_score", "mean_test_score"]).head(10)

# %%
"""
##### Bayesian search
"""

# %%
"""
> there is a conflict with new versions of numpy in skopt (np.int is deprecated in new versions),  
and it's not fixed, for fixing it we will use a trick to fix it.
```python
np.int = np.int_
```
"""

# %%
# using Bayesian search
np.int = np.int_

search_spaces = {
    "n_estimators": [20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 500, 600],
    "criterion": ['gini', 'entropy', 'log_loss'],
    "min_samples_split": [2, 4, 6, 8, 10, 12, 20],
    "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15],
    "max_depth": [50,70, 100, 150, 200, 300],
}


bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=search_spaces,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=5,

)


bayes_search.fit(X_train, y_train)

# %%
"""
**it was super fast!!!, although we used more parameters than the other methods!**
"""

# %%
bayes_search.best_score_

# %%
bayes_search.best_params_

# %%
pd.DataFrame(bayes_search.cv_results_).sort_values(by=["rank_test_score", "mean_test_score"]).head(10)

# %%
"""
<br> <br> <br>
___
<br> <br> <br>
"""

# %%
"""
there is also another way you can use `BayesSearchCV`
"""

# %%
# using Bayesian search
np.int = np.int_
from skopt.space import Integer

search_spaces = {
    "n_estimators": Integer(50, 200),
    "criterion": ['gini', 'entropy', 'log_loss'],
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
    "max_depth": Integer(50, 300),
}


bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=search_spaces,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=5,

)


bayes_search.fit(X_train, y_train)

# %%
bayes_search.best_params_

# %%
bayes_search.best_score_

# %%
pd.DataFrame(bayes_search.cv_results_).sort_values(by=["rank_test_score", "mean_test_score"]).head(5)

# %%
"""
#### HyperParameter tuning for XGBoost
"""

# %%
"""
##### Bayesian Search
"""

# %%
search_spaces = {
    'n_estimators': (50, 200),
    'learning_rate': (0.01, 0.3, 'uniform'),
    'max_depth': (3, 50),
    'subsample': (0.5, 1.0, 'uniform'),
    'colsample_bytree': (0.5, 1.0, 'uniform'),
    'gamma': (0, 1, 'uniform'),
    'min_child_weight': (1, 10),
    'reg_alpha': (0, 1, 'uniform'),
    'reg_lambda': (0, 1, 'uniform')
}

xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=1502,
    n_jobs=-1,
)

xgboost_bayes_search = BayesSearchCV(
    estimator=xgb_classifier,
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    verbose=5,
    n_jobs=-1  # Use all available CPU cores
)

xgboost_bayes_search.fit(X_train, y_train)

# %%
xgboost_bayes_search.best_score_

# %%
xgboost_bayes_search.best_params_

# %%
"""
# Evaluation
"""

# %%
"""
#### Logistic Regression
"""

# %%
# confusion matrix for 0.99 percent model
X = MinMax_scaled_df.drop(columns="class")
y = MinMax_scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

selected_features = corr.query("correlation >= 0.5 | correlation <= -0.5").index
lr = LogisticRegression()
lr.fit(X_train[selected_features], y_train)

matrix = confusion_matrix(y_test, lr.predict(X_test[selected_features]))
matrix = pd.DataFrame(
    matrix,
    columns=["Positive", "Negative"], # true values
    index=["Positive", "Negative"],   # predicted values
)
matrix

# %%
# cooler version :)
matrix = matrix.apply(
    lambda row: row / np.sum(row, axis=0),
    axis=1
)

# palette = sns.cubehelix_palette(as_cmap=True)
ax = sns.heatmap(
    matrix,
    annot=True, 
    cbar=False,
    center=0.3,
    cmap="Blues",
)

plt.yticks(rotation=0);

# %%
"""
#### Decision Tree
"""

# %%
X = scaled_df[selected_features] # `class` feature is not here
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

matrix = confusion_matrix(y_test, dt.predict(X_test))
matrix = pd.DataFrame(
    matrix,
    columns=["Positive", "Negative"], # true values
    index=["Positive", "Negative"],   # predicted values
)
matrix

# %%
matrix = matrix.apply(
    lambda row: row / np.sum(row, axis=0),
    axis=1
)
ax = sns.heatmap(
    matrix,
    annot=True, 
    cbar=False,
    center=0.3,
    cmap="Blues",
)

plt.yticks(rotation=0);

# %%
"""
#### Random Forest
"""

# %%
"""
for this we are going to ues `RandomizedSearchCV` !!
"""

# %%

X = scaled_df[selected_features] # `class` feature is not here
y = scaled_df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1502)

params = {
    'n_estimators': [5, 10, 20, 30, 50, 100],
    'max_depth': [2, 4, 6, 8],
    'bootstrap': [True, False],
}

rscv = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=params,
    n_iter=20,
    cv=5,
)

rscv.fit(X_train, y_train)
rscv.best_params_

# %%
rscv.best_score_

# %%
matrix = confusion_matrix(y_test, rscv.predict(X_test))
matrix = pd.DataFrame(
    matrix,
    columns=["Positive", "Negative"], # true values
    index=["Positive", "Negative"],   # predicted values
)
matrix

# %%
matrix = matrix.apply(
    lambda row: row / np.sum(row, axis=0),
    axis=1
)
ax = sns.heatmap(
    matrix,
    annot=True, 
    cbar=False,
    center=0.3,
    cmap="Blues",
)

plt.yticks(rotation=0);
#%%
import os
import pandas as pd


#%%
covid_19 = pd.read_csv("covidData.csv", encoding= 'unicode_escape')
covid_19.describe()
covid_19.info()
covid_19.dropna()

# %%
X = covid_19.drop(columns=['Patient_ID','DT_Collected','COVID PCR Result'])
y = covid_19['COVID PCR Result']



# %% Checking which columns are objects
X.columns[X.dtypes.eq('object')]

# %% We convert Age and HGB to numeric
cols = X.columns.drop('Patient_Sex') # select all columns except Patient_sex
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce') # Convert all to numeric

#%% One-hot encoding for gender
X = pd.get_dummies(X, columns = ['Patient_Sex'])

#%%
X = X.fillna(X.median())

#%% Tackling class imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X,y = oversample.fit_resample(X, y)

# %% Partitioning data into train and test 80-20
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X,y, test_size = .2, random_state= 123)

# %% Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfModel = RandomForestClassifier(n_estimators=1000, random_state= 123)
rfModel.fit(trainX,trainY)

# %% Making predictions
prediction = rfModel.predict(testX); prediction

# %% Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(testY, prediction)

#%%
accuracy_score(testY, prediction)

# %%
from matplotlib import pyplot as plt
sort_index = rfModel.feature_importances_.argsort()
plt.barh(X.columns[sort_index], rfModel.feature_importances_[sort_index])
plt.xlabel("Random Forest Feature importance")
# %%



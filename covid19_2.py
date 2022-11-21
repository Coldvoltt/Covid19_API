#%%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

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

#%% The model fn
def train(df_train, y_train):

    df_train = df_train.to_numpy()

    # Random Forest model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=1000, random_state= 123)
    model.fit(df_train,y_train)

    return model

# Prediction function
def predict(df, model):

    df = df.to_numpy()
    y_pred = model.predict(df)
    return y_pred


#%%
model = train(trainX, trainY)

# %% Prediction and accuracy
pred_y = predict(testX, model)
accuracy_score(testY, pred_y)


# %% Saving the model
import pickle
pickle.dump(model,open('rfModel.pkl', 'wb'))


#%% Sample individual data
patient = testX.iloc[3].to_numpy()

#%%
rfMod = pickle.load(open('rfModel.pkl', 'rb'))

# %%
rfMod.predict(patient.reshape(1,-1))

#%% Probability
model.predict_proba(patient)[0,1]

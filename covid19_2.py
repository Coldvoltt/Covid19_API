#%%
import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
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
def trn(df_train, y_train):
    # Dictionary vectorizer
    dicts = df_train.to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # Random Forest model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=1000, random_state= 123)
    model.fit(X_train,y_train)

    return dv, model

# Prediction function
def pred(df, dv, model):
    dicts = df.to_dict(orient = 'records')
    X = dv.transform(dicts)

    y_pred = model.predict(X)
    return y_pred


#%%
dv, model = trn(trainX, trainY)

# %% Prediction and accuracy
pred_y = pred(testX, dv, model)
accuracy_score(testY, pred_y)


# %% Saving the model
import pickle
output_file = f'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


#%% Sample individual data
patient = testX.iloc[3].to_dict()
# %%
patient = dv.transform(patient)
model.predict(patient)
# %%

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
def train(df_train, y_train):
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
def predict(df, dv, model):
    dicts = df.to_dict(orient = 'records')
    X = dv.transform(dicts)

    y_pred = model.predict(X)
    return y_pred


#%%
dv, model = train(trainX, trainY)

# %% Prediction and accuracy
pred_y = predict(testX, dv, model)
accuracy_score(testY, pred_y)


# %% Saving the model
import pickle
output_file = f'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


#%% Sample individual data
patient = testX.iloc[3].to_dict()

# %%
X_dv = dv.transform(patient)
model.predict(X_dv)

#%% Probability
model.predict_proba(X_dv)[0,1]


# %%
# Creating Patient's blood test result as input
ipatient = {'Age': 53.0,
    'Hematocrit': 41.7,
    'HCT 35-45%': 1.0,
    'HGB': 14.3,
    'HGB 12-15.5 g/dl': 0.0,
    'MCH': 29.7,
    'MCH 26-34 pg': 0.0,
    'MCHC': 34.3,
    'MCHC 31-36 g/dl': 0.0,
    'MCV': 86.5,
    'MCV 82-98 fl': 0.0,
    'RBC Count': 4.82,
    'RBC Count 3.9 - 5.0 x10^6/uL': 0.0,
    'RDW': 12.2,
    'RDW 11.5 - 16.5%': 0.0,
    'Total WBC Count': 4.31,
    'WBC Count 3.5-10.5 x10^3/uL': 0.0,
    'Lymphocyte ': 1500.0,
    'Lymphocyte 900 - 2900 ?': 0.0,
    'Basophils ': 30.0,
    'Basophil 0 - 100 ?': 0.0,
    'Eosinphils ': 5.1,
    'Eosinphils 50 - 500 uL': 1.0,
    'Neutrophil ': 2142.0,
    'Neutrophils 1700 - 8000 ?': 0.0,
    'Monocytes ': 409.0,
    'Monocytes 300 - 900 ?': 0.0,
    'Plt Counts': 225.0,
    'Platelet 150-450 x10^3/uL   ': 0.0,
    'Patient_Sex_F': 0.0,
    'Patient_Sex_M': 1.0
    }
# %%

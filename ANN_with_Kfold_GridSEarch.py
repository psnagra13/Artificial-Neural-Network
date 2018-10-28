####  ANN : Artificial Neural Network using Keras
####  This is a template for Classification task


### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from keras.layers import Dropout

### Data Preprocessing 
dataset = pd.read_csv('datasets/Churn_Modelling.csv')   # Load Dataset from file

X = dataset.iloc[:, 3:13].values     # Choose Feature coloumns
y = dataset.iloc[:, 13].values        # Choose label coloumn


## Encoding categorical coloumns
# Encode for coloumn 1 (numbering starts from 0,1,2,3....)
labelEncoderX1 = LabelEncoder()
X[:,1] = labelEncoderX1.fit_transform(X[:,1])
# Encode for coloumn 2 (numbering starts from 0,1,2,3....)
labelEncoderX2 = LabelEncoder()
X[:,2] = labelEncoderX2.fit_transform(X[:,2])
# One Hot Encoding for coloumn 1
oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Defining NN as sequence of layers
## Initializing ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu', input_dim= 11 ))
    classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu' ))
    classifier.add( Dense(output_dim = 1, init = 'uniform', activation= 'sigmoid' ))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy' , metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier  )
parameters = {
                'batch_size': [25,32],
                'nb_epoch' : [10,50],
                'optimizer' : ['adam', 'rmsprop']
            }

grid_search = GridSearchCV( estimator = classifier,
                            param_grid=parameters,
                            scoring ='accuracy',
                            cv= 10)

grid_search = grid_search.fit(X_train,y_train)
 
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
 


















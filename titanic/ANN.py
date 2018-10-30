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
dataset = pd.read_csv('train.csv')   # Load Dataset from file
dataset['Age'] = dataset['Age'].fillna(30)
dataset['Embarked'] = dataset['Embarked'].fillna('S')

X = dataset.iloc[:, [2,4,5,6,7,9,11]].values     # Choose Feature coloumns
y = dataset.iloc[:, 1].values        # Choose label coloumn


## Encoding categorical coloumns
# Encode for coloumn 1 (numbering starts from 0,1,2,3....)
labelEncoderX1 = LabelEncoder()
X[:,1] = labelEncoderX1.fit_transform(X[:,1])

# Encode for coloumn 1 (numbering starts from 0,1,2,3....)
labelEncoderX2 = LabelEncoder()
X[:,6] = labelEncoderX2.fit_transform(X[:,6])

oneHotEncoder = OneHotEncoder(categorical_features=[6]   )
X = oneHotEncoder.fit_transform(X).toarray()

X = X[:, 1:]


df = pd.DataFrame(X)

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
classifier = Sequential()

## Adding Layers to ANN
# Adding Input Layer and First Hiddent Layer
classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu', input_dim= 8 ))
#            ouput_dim = number Of Neurons In This Layer                input_dim = number of neurons in previous layer (here previous layer is input layer)
classifier.add(Dropout(0.1))

# Add 2nd hidden layer
classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu' ))
classifier.add(Dropout(0.1))

# Add 3rd hidden layer
classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu' ))
classifier.add(Dropout(0.1))

# Add Output layer
classifier.add( Dense(output_dim = 1, init = 'uniform', activation= 'sigmoid' ))
#                            output_dim - here it is 1, because it is binary classifier(only 1 class)                  
#                               If the output nodes> 1; that is there are more classes, use activation = "softmax"

# Compiling  ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy' , metrics = ['accuracy'])


# Fitting the ANN
classifier.fit( X_train , y_train , batch_size= 10 , nb_epoch =100)

# Make Predictions on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.4)



# Making COnfusion Matrix
cm = confusion_matrix(y_test, y_pred)













### Data Preprocessing 
dataset_pred = pd.read_csv('test.csv')   # Load Dataset from file
dataset_pred['Age'] = dataset_pred['Age'].fillna(30)
dataset_pred['Fare'] = dataset_pred['Fare'].fillna(25)

dataset_pred['Embarked'] = dataset_pred['Embarked'].fillna('S')

X = dataset_pred.iloc[:, [1,3,4,5,6,8,10]].values     # Choose Feature coloumns


## Encoding categorical coloumns
# Encode for coloumn 1 (numbering starts from 0,1,2,3....)
X[:,1] = labelEncoderX1.transform(X[:,1])

# Encode for coloumn 1 (numbering starts from 0,1,2,3....)
X[:,6] = labelEncoderX2.transform(X[:,6])


X = oneHotEncoder.transform(X).toarray()

X = X[:, 1:]


df = pd.DataFrame(X)


X = sc.transform(X)


# Make Predictions on test set
y_pred = classifier.predict(X)
y_pred = (y_pred > 0.4).astype(int)

dataset_pred['Survived'] = y_pred




dataset_pred.to_csv('gender_submission.csv', sep=',')

















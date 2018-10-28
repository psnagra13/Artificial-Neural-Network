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
classifier = Sequential()

## Adding Layers to ANN
# Adding Input Layer and First Hiddent Layer
classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu', input_dim= 11 ))
#            ouput_dim = number Of Neurons In This Layer                input_dim = number of neurons in previous layer (here previous layer is input layer)

# Add 2nd hidden layer
classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu' ))

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
y_pred = (y_pred > 0.5)



# Making COnfusion Matrix
cm = confusion_matrix(y_test, y_pred)


#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000

new_data = np.array([[0.0,0,619,  1, 40 , 3, 60000 , 2 , 1, 1, 50000 ]])

new_data = sc.transform(new_data)
new_prediction = classifier.predict(new_data)
new_prediction = (new_prediction >0.5)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu', input_dim= 11 ))
    classifier.add( Dense(output_dim = 6, init = 'uniform', activation= 'relu' ))
    classifier.add( Dense(output_dim = 1, init = 'uniform', activation= 'sigmoid' ))
    classifier.compile(optimizer='adam', loss='binary_crossentropy' , metrics = ['accuracy'])
    return classifier
    
classifier_Kfold = KerasClassifier(build_fn = build_classifier , batch_size= 10 , nb_epoch =10 )
accuracies = cross_val_score(estimator= classifier_Kfold , X= X_train , y= y_train, cv=10, n_jobs=-1 )




mean = accuracies.mean()














import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X_1 = LabelEncoder()
X[:,1]=label_encoder_X_1.fit_transform(X[:,1])

label_encoder_X_2 = LabelEncoder()
X[:,2] = label_encoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

# Adding the inpout layer and the first hidden layer with dropout
classifier.add(Dense(output_dim = 6, init='uniform',activation='relu', input_dim=11))
classifier.add(Dropout(p=0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim =1, init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch=100)

y_pred = classifier.predict(X_test)
y_pred =(y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting a new observation
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)

# Part 4: Evaluating imprtoving and Tuning the ANN

# Evaluating the ANN


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init='uniform',activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform',activation='relu'))
    classifier.add(Dense(output_dim =1, init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN


# Tuning The ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init='uniform',activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform',activation='relu'))
    classifier.add(Dense(output_dim =1, init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss ='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters ={'batch_size':[25, 32],
             'nb_epoch',[100,500],
             'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
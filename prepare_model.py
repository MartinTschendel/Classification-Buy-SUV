#Build Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#load dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#build the model
clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#train classifier
clf.fit(X_train, y_train)

#predictions
predicted = clf.predict(X_test)

#check accuracy
print(accuracy_score(predicted, y_test))

#pickel the model
#save the python object to a binary file
#afterwards we can reuse it
#we do not have to train the model again
import pickle
#'wb' means write binary
with open ('rf.pkl', 'wb') as model_pkl:
    pickle.dump(clf, model_pkl)

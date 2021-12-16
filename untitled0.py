

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Importing the dataset
dataset = pd.read_csv('LDR.csv')
X = dataset.iloc[:,0:2].values

y = dataset.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
# Splitting the dataset into the Training set and Test set

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
y[0:] = labelencoder_X.fit_transform(y[0:])

X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5)

y_pred = kmeans.fit_predict(X_test)
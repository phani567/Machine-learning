import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus =[]
for i in range(0,1000):
    review =re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #removing all except letters
    review =review.lower() #change to lowercase
    review =review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #removing the stopwords like articles and stemming the words
    review = ' '.join(review)
    corpus.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X =cv.fit_transform(corpus).toarray()
y =dataset.iloc[:,1].values

#applying machine learning model
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting the logistic Regression Model to the dataset
from sklearn import linear_model
classifier =linear_model.LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classifier.score(X_test,y_test))
print(classifier.score(X_train,y_train))
precision =(cm[0][0]/(cm[0][0]+cm[0][1])) 
Recall =(cm[0][0]/(cm[0][0]+cm[1][0]))
print ("precision :",precision)
print("recall: " ,Recall)
print("F1 score: ",(2*precision*Recall/(precision+Recall)))

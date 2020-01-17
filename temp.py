# Natural Language Processing

# Importing the libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Importing the dataset
col_names = ['Rating', 'Review']
dataset = pd.read_csv('data/train.csv', names=col_names, header=None)
del col_names


# Cleaning the texts
corpus = []
for i in range(0,50000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:len(X), 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Creating the Bag of Words model
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:len(X), 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, random_state = 0)


# Fitting Naive Bayes to the Training set
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)

# Predicting the Test set results
y_predNB = classifierNB.predict(X_val)



#Check Accuracy :

a = 0
for i in range(len(y_predNB)):
    if y_predNB[i]==y_val[i]:
        a = a + 1
accuracyNB = (a/len(y_predNB))*100
# Natural Language Processing

# Importing the libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding
from nltk.stem import SnowballStemmer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split




# Importing the dataset
col_names = ['Rating', 'Review']
dataset = pd.read_csv('data/train.csv', names=col_names, header=None)
del col_names

#======================================================================================================
# Cleaning the texts

corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = SnowballStemmer('english')
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
del review,i

#======================================================================================================


X = corpus
y = dataset.iloc[0:len(corpus), 0].values
#X_train, X_val, y_train, y_val = train_test_split(corpus, y, test_size = 0.10, random_state = 0)

def decrement(list):
    return [x - 1 for x in list]
#Using Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
vocabulary_size = len(tokenizer.word_index) + 1
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=100,padding="post")

num_classes = 5
y = to_categorical(decrement(y), num_classes=num_classes)

#======================================================================================================
#Using CNN over LSTM


model_conv = Sequential()
model_conv.add(Embedding(vocabulary_size, 100, input_length=100))
model_conv.add(Dropout(0.2))
model_conv.add(Conv1D(64, 5, activation='relu'))
model_conv.add(MaxPooling1D(pool_size=4))
model_conv.add(LSTM(100))
model_conv.add(Dense(5, activation='softmax'))
model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_conv.fit(X, y, batch_size=64, epochs=10, validation_split = 0.2)
#======================================================================================================
#Using LSTM


model = Sequential()
model.add(Embedding(vocabulary_size, 100, input_length=100))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, batch_size=64, epochs=10, validation_split = 0.2)



#======================================================================================================
#Using Naive Bayes Classification

cv = CountVectorizer(max_features = 1000)
X_nb = cv.fit_transform(corpus).toarray()
y_nb = dataset.iloc[0:len(corpus), 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X_nb, y_nb, test_size = 0.20)

#Classification
#Fitting Naive Bayes to the Training set
classifierNB = GaussianNB()
classifierNB.fit(X, y, epochs=10, validation_split = 0.2)

## Predicting the Test set results
y_predNB = classifierNB.predict(X_val)
#
#Check Accuracy :
a = 0
for i in range(len(y_predNB)):
    if y_predNB[i]==y_val[i]:
        a = a + 1
accuracyNB = (a/len(y_predNB))*100
del a
print('Accuracy is : ', accuracyNB)

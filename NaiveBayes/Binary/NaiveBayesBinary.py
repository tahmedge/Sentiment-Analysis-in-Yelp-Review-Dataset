import numpy as np
from nltk.stem import *
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
ps = PorterStemmer()
#read the train file
a = open("train2star.txt", "r",encoding="utf-8")
b = open("valid2star.txt", "r",encoding="utf-8")
c = open("test2star.txt", "r",encoding="utf-8")

x_train=[]
y_train=[]
j=0

tempstr=[]
for i in a:
   j=j+1
   #print(i)
   if(j%2==1):
        ii = i.split()
        for w in ii:
            temp=w
            tempstr.append(temp)
        tempstr=(' ').join(tempstr)
        x_train.append(tempstr)
   else:
        y_train.append(int(i))
   tempstr=[]

j=0
for i in b:
   j=j+1
   #print(i)
   if(j%2==1):
        ii = i.split()
        for w in ii:
            temp=w
            tempstr.append(temp)
        tempstr=(' ').join(tempstr)
        x_train.append(tempstr)
   else:
        y_train.append(int(i))
   tempstr=[]

#print(len(y_train))
x_test=[]
y_test=[]
tempstr = []
j=0
for i in c:
   j=j+1
   if(j%2==1):
        ii = i.split()
        for w in ii:
            temp=w
            tempstr.append(temp)
        tempstr=(' ').join(tempstr)
        x_test.append(tempstr)
   else:
       y_test.append(int(i))
   tempstr=[]

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(x_train)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

clf = MultinomialNB().fit(train_tfidf, y_train)

test_counts = count_vect.transform(x_test)
test_tfidf = tfidf_transformer.transform(test_counts)
predicted = clf.predict(test_tfidf)
score = metrics.accuracy_score(y_test,predicted)
print('%.1f%%' %(score*100))


#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()
#clf.fit(x_train, y_train)

#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

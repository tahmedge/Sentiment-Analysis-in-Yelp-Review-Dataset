import numpy as np
from nltk.stem import *
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Bidirectional
import numpy as np
from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
ps = PorterStemmer()
#read the train file
a = open("train2Star.txt", "r",encoding="utf-8")
b = open("valid2Star.txt", "r",encoding="utf-8")
c = open("test2Star.txt", "r",encoding="utf-8")

x_train=[]
y_train=[]
j=0

tempstr=[]
for i in a:
   j=j+1

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


x_valid=[]
y_valid=[]
j=0

tempstr=[]
for i in b:
   j=j+1

   if(j%2==1):
        ii = i.split()
        for w in ii:
            temp=w
            tempstr.append(temp)
        tempstr=(' ').join(tempstr)
        x_valid.append(tempstr)
   else:
        y_valid.append(int(i))
   tempstr=[]






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


tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(sequences, maxlen=300)

embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

#tokenizer.fit_on_texts(x_test)
sequences = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(sequences, maxlen=300)

#tokenizer.fit_on_texts(x_valid)
sequences = tokenizer.texts_to_sequences(x_valid)
x_valid = pad_sequences(sequences, maxlen=300)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=2)
model = Sequential()
model.add(Embedding(20000, 100, input_length=300))
#Use the following line if you want to use GloVe word embedding. Comment the previous line before using GloVe.
#model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=300, trainable=False))
model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history=model.fit(x_train,np.array(y_train),validation_data=(x_valid,np.array(y_valid)), epochs=10, callbacks=[early_stopping])
scores = model.evaluate(x_test,(np.array(y_test)))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

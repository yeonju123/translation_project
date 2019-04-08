#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:37:40 2019

@author: yeonjulee
"""
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed
from keras.models import load_model
from sklearn.utils import shuffle
from keras import optimizers
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 200)

def read_text(filename):
    file = open(filename, mode = "rt", encoding = "utf-8")
    text = file.read()
    file.close()
    return text


def to_lines(text):
    sents = text.split("\n")
    sents = [i.split("\t") for i in sents]
    return sents

data = read_text("/users/yeonjulee/Downloads/en-fa.txt/OpenSubtitles.en-fa.en")
en = to_lines(data)
en = array(en)

en[1]

data = read_text("/users/yeonjulee/Downloads/en-fa.txt/OpenSubtitles.en-fa.fa")
fa = to_lines(data)
fa = array(fa)
fa[1]

en_fa = pd.DataFrame(en, columns = ["ENG"])
en_fa.head()
en_fa["PERR"] = fa

en_fa["ENG"] = en_fa["ENG"].str.lower()

en_fa.head()
en_fa["PERR"] = en_fa["PERR"].str.lower()

eng_l = []
fa_l = []

for i in en_fa["ENG"]:
    eng_l.append(len(i.split()))

for i in en_fa["PERR"]:
    fa_l.append(len(i.split()))
    

en_fa["ENG_L"] = array(eng_l)
en_fa["PERR_L"] = array(fa_l)


en_fa = shuffle(en_fa)

# data size is defined
len(en_fa)
data_size = 50000
train_size = int(data_size * 0.8)
en_fa.iloc[0]

data_docs_cut = en_fa.iloc[:data_size +1]  #cut data into 10000

train_docs = data_docs_cut[:train_size]
test_docs = data_docs_cut[train_size:]

def max_length(lines):
    return max(len(line.split()) for line in lines)

def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

#encoder 
def encode_sequences(tokenizer,length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen = length, padding = "post")
    return seq

# this part has to be fixed re-shaping
#def encode_output(sequences, vocab_size):
 #   ylist = list()
  #  for sequence in sequences:
   #     encoded = to_categorical(sequence, num_classes = vocab_size)
    #    ylist.append(encoded)
     #   y = array(ylist)
      #  y = y.reshape(sequences.shape[0], sequences.shape[1])
       # return y
    
# this part has to be fixed re-shaping
        
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length = in_timesteps,
                        mask_zero = True))
    model.add(LSTM(n))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences = True))
    model.add(TimeDistributed(Dense(out_vocab, activation = "softmax")))
    return model


eng_tokenizer = tokenization(train_docs["ENG"])
eng_vocab_size = len(eng_tokenizer.word_index)+1
fa_tokenizer = tokenization(train_docs["PERR"])
fa_vocab_size = len(fa_tokenizer.word_index)+1
eng_length = max_length(train_docs["ENG"]) #length of the longest sentence token
fa_length = max_length(train_docs["PERR"])


# this part has to be fixed re-shaping


trainX = encode_sequences(eng_tokenizer, eng_length, train_docs["ENG"])
trainY = encode_sequences(fa_tokenizer, fa_length, train_docs["PERR"])
#reshpaing is not working now =====================================

# this part has to be fixed re-shaping




testX = encode_sequences(eng_tokenizer, eng_length, test_docs["ENG"])
testY = encode_sequences(fa_tokenizer, fa_length, test_docs["PERR"])




model = build_model(eng_vocab_size, fa_vocab_size, eng_length, fa_length, 512 ) #512 hidden units 
rms = optimizers.RMSprop(lr = 0.001)

model.compile(optimizer = rms, loss = "sparse_categorical_crossentropy")
#model.compile(optimizer = "adam", loss = "categorical_crossentropy")
filename = "model.en_fa_translater"

print(model.summary())


#set checkpoint
checkpoint = ModelCheckpoint(filename, monitor = "val_loss",
                             verbose = 1, save_best_only = True,
                             mode = "min")

#traub model 
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1],  1), # decoder is intergraded here
                    epochs = 30, batch_size = 512, validation_split = 0.2,
                    callbacks = [checkpoint], verbose = 1)



plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train","validation"])
plt.show()

model = load_model("model.en_fa_translater")
preds = model.predict_classes(testX.reshape((testX.shape[0], testX.shape[1])))

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

#convert predictions into text

    

preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], fa_tokenizer)
        if j > 0:
            if (t == get_word(i[j]-1, fa_tokenizer)) or (t == None):
                temp.append("")
            else:
                temp.append(t)
        else:
            if(t == None):
                temp.append("")
            else:
                temp.append(t)
    preds_text.append(" ".join(temp))
    
    
pred_df = pd.DataFrame({"actual": test_docs["PERR"], "predicted": preds_text})

#print 15 rows randomly
pred_df.tail(30)
pred_df["actual"].tail()

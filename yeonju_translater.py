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

# load parallel corpus
def read_text(filename):
    file = open(filename, mode = "rt", encoding = "utf-8")
    text = file.read()
    file.close()
    return text


def to_lines(text):
    sents = text.split("\n")
    sents = [i.split("\t") for i in sents]
    return sents

data = read_text("/home/compling4/Desktop/ANLP_project-master/en-ko.txt/OpenSubtitles.en-ko.en")
en = to_lines(data)
en = array(en)

en[1]

data = read_text("/home/compling4/Desktop/ANLP_project-master/en-ko.txt/OpenSubtitles.en-ko.ko")
target = to_lines(data)
target = array(target)
target[1]

# put each language file into a dataframe in a parallel way, separated by line.

en_ta = pd.DataFrame(en, columns = ["ENG"])
en_ta.head()
en_ta["Target"] = target

en_ta["ENG"] = en_ta["ENG"].str.lower()

en_ta.head()
en_ta["Target"] = en_ta["Target"].str.lower()


#get the length of each line for each language
eng_l = []
ta_l = []

for i in en_ta["ENG"]:
    eng_l.append(len(i.split()))

for i in en_ta["Target"]:
    ta_l.append(len(i.split()))
    

en_ta["ENG_L"] = array(eng_l)
en_ta["Target_L"] = array(ta_l)

# shuffle data
en_ta = shuffle(en_ta)

# data size is defined
len(en_ta)
data_size = 50000
train_size = int(data_size * 0.8)
en_ta.iloc[0]

data_docs_cut = en_ta.iloc[:data_size +1]  #cut data into 10000

train_docs = data_docs_cut[:train_size]
test_docs = data_docs_cut[train_size:]

def max_length(lines):
    return max(len(line.split()) for line in lines)

def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

#encoder, vectorizing words with one-hot representation. Here, tokenizer is used to vectorize.
def encode_sequences(tokenizer,length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen = length, padding = "post")
    return seq


#build a model

def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length = in_timesteps,
                        mask_zero = True))
    model.add(LSTM(n))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences = True))
    model.add(TimeDistributed(Dense(out_vocab, activation = "softmax")))
    return model

# vectorizing both English and Target language of traning set.
eng_tokenizer = tokenization(train_docs["ENG"])
eng_vocab_size = len(eng_tokenizer.word_index)+1
ta_tokenizer = tokenization(train_docs["Target"])
ta_vocab_size = len(ta_tokenizer.word_index)+1
eng_length = max_length(train_docs["ENG"]) #length of the longest sentence token
ta_length = max_length(train_docs["Target"])


# this part has to be fixed re-shaping


trainX = encode_sequences(eng_tokenizer, eng_length, train_docs["ENG"])
trainY = encode_sequences(ta_tokenizer, ta_length, train_docs["Target"])
#reshpaing is not working now =====================================

# this part has to be fixed re-shaping




testX = encode_sequences(eng_tokenizer, eng_length, test_docs["ENG"])
testY = encode_sequences(ta_tokenizer, ta_length, test_docs["Target"])


model = build_model(eng_vocab_size, ta_vocab_size, eng_length, ta_length, 512 ) #512 hidden units 
rms = optimizers.RMSprop(lr = 0.001)

model.compile(optimizer = rms, loss = "sparse_categorical_crossentropy")
#model.compile(optimizer = "adam", loss = "categorical_crossentropy")
filename = "model.en_target_translater"

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

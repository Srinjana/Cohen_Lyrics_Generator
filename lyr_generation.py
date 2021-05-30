# Traininng a Neural Network to generate new Leonard Cohen ❤ Lyrics using Recurrent Neural Networks.
# Uses Long Short term Memory (LSTM) to predict the next probable word based on previous trends and vice versa.
#we use a bidirectional LSTM on the training corpus to achieve required results.

#IMPORTING NECESSARY PACKAGES

import tensorflow as tf
import matplotlib as plt
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

#READING THE INPUT DATASET

data = open('leonard_cohen.txt').read()
corpus = data.lower().split("\n")

#PREPROCESSING

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1   #adding 1 for oov token

# print(tokenizer.word_index)
# print(total_words)

#CREATING TRAINING DATA

input_sequences = []

for line in corpus:
    #creating list of tokens for every line in corpus line by line

    token_list = tokenizer.texts_to_sequences([line])[0]

    #generating n-grams ("you see these, this is what comes next" word by word)

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
    
#PADDING SEQUENCES
max_sequence_len = max([len(x) for x in input_sequences]) 
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding = 'pre'))   

#ADDING FEATURES AND LABELS.

#in the padded sequences everything but the last value is an x and the last value foms the label y.
#E.g. [0 0 0 0 0 3 4 5]  {5 is a label and everything else is under x(feature)}
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

#converts class vector(labels) to a binary class matrix
ys = tf.keras.utils.to_categorical(labels, num_classes = total_words)

#TRAINING THE NEURAL NETWORK
model = Sequential()
model.add(Embedding(total_words, 240, input_length = max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation = 'softmax'))
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs = 100, verbose=1)

#Checking Accuracy (HOUSE-KEEPING)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title("Accuracy Index")
    plt.show()

plot_graphs(history, 'accuracy')

#GENERATING NEW TEXT

#seed_text = input()
#ideally the seed_text should be user input, it's cute, but this sppeds things up a bit.

seed_text = 'I had you in my arms'
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen= max_sequence_len - 1, padding = 'pre')
    predicted = model.predict_classes(token_list, verbose = 0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    print(seed_text)
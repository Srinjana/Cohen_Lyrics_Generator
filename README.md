# Leonard Cohen Lyrics Generator
### This project uses Deep Learning techniques such as Natural Language Processing (NLP), Dense RNN and a Bidirectional LSTM to create an entirely AI-generated, Leonard Cohen song.
---

The [Original Dataset](https://www.kaggle.com/paultimothymooney/poetry?select=leonard-cohen.txt) used for this purpose is by [Paul Mooney](https://www.kaggle.com/paultimothymooney).
Although I added a few more of his lyrics to it. The updated dataset is available as `leonard_cohen.txt`

## Working Concepts:

### This diagram explains the basic concept of "HOW RNN WORKS ?"

![UNDERSTAND RNN WITH THIS FIGURE](https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/10/1-33-768x287.png)

---

### Here I will do a thorough run-down of all the steps I followed to reach my desired output:~
* Firstly the necessary packages are imported and the training dataset is read line by line.
* Each word in the line is assigned an index. This index is known as a <b> `Token` </b>. Every new word that appears is assigned a new token.
* An extra token is also assigned for Out of Vocabulary words.
* Each sentence is then Expressed as a list of Tokens.
* The Tokens are Padded, so that all the lists are of same size, so 0s are added in front of the token list to make it fit the largest list size.
* [`N-grams`](https://kavita-ganesan.com/what-are-n-grams/#.YLNmdKgzZPY) are generated word by word for each line in the `corpus`, to train the neural network about the next word to probably follow.
* Predictors and Labels are fitted to the padded corpus to prepare it for training.
* The Model is Embedded with a Dense Neural Network and a Bidirectional LSTM is used in order to be able to generate the next probable word.
* Loses are hot encoded using `categorical-crossentropy` and `Adam` Optimizer is used.
* The model fits the Xs and Ys for Hundred epochs and checks the `accuracy` metric.

 ![](ss/epochs.png)

* A graph is plotted to obtain the rate of accuracy.
### Rate of Accuracy : 69-70%

![](ss/graphs.png)

* After training the Neural Network is fed with some Seed Text and is allowed to generate its own lyrics upto Hundred next words.

## Input Fed:
```
"You were my reason"
```
## Output Lyrics:
```

" You were my reason
it was before
it was before
it was before
going to do my lip
you're all to the last
you're of song i know
it was trouble an altar shivering shivering around into into his skin
of us turned well on go crazy our kisses too became
been dying spark sand eye around from her mirror twin soon every night
that strong soon must be back on again about on the before
soon be sailors into his fiery flare and her eye around school
disregard her mirror and death by surprise them
here car has been working out of world "

```
I'd say that's not half as bad as I thought it'd be. :)

## Necessary Packages:
```
import tensorflow as tf
import matplotlib as plt
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
```
## Files:

* `lyr_generation.py   =  PYTHON FILE`
* `lyric_gen.ipynb    = JUPYTER NOTEBOOK FILE`

---
Find the video Explaining the technologies used [here](https://www.youtube.com/watch?v=ZMudJXhsUpY)

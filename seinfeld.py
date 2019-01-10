import tensorflow as tf
import pandas as pd
import nltk
import math
import itertools
import numpy as np

seinfeld_data = pd.read_csv('Sentence_generation/scripts.csv')

seinfeld_text_data = seinfeld_data[['Dialogue']]

VOCAB_SIZE = 5000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64

sentences =  []
for i in range(0, len(seinfeld_text_data)):
    temp = seinfeld_text_data.iloc[i]['Dialogue']
    if type(temp) is str:
          
        temp = nltk.sent_tokenize(temp)
    
        for j in range(0, len(temp)):
            temp[j] = sentence_start_token + ' ' + temp[j] + ' ' + sentence_end_token 
            sentences.append(temp[j])

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

vocab = word_freq.most_common(VOCAB_SIZE-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

training_data = []
WINDOW_SIZE = 2

for sentence in tokenized_sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                training_data.append([word, nb_word])

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

oneHot_X_train = []
oneHot_Y_train = []

for data_word in training_data:
    oneHot_X_train.append(to_one_hot(word_to_index[ data_word[0] ], VOCAB_SIZE))
    oneHot_Y_train.append(to_one_hot(word_to_index[ data_word[1] ], VOCAB_SIZE))

print(oneHot_X_train)



import json
raw_json = json.load(open("courseraDeepAI/resources/sarcasm.json"))
actual_comments=[]
labels=[]

for each in raw_json:
  actual_comments.append(each["headline"])
  labels.append(each["is_sarcastic"])


print(len(actual_comments))
print(len(labels))
print(actual_comments[0])
print(labels[0])

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences


my_token = Tokenizer(10000,oov_token="<oov_token>")
my_token.fit_on_texts(actual_comments)

seq=my_token.texts_to_sequences(actual_comments)
token_index = my_token.word_index

max([len(x) for x in seq]),min([len(x) for x in seq])

padded_seq = pad_sequences(seq,padding='post')


total_len  = len(padded_seq)
train_token = padded_seq[:int(0.9*total_len)]
train_label = labels[:int(0.9*total_len)]
print(len(train_token))
print(len(train_label))

valid_token = padded_seq[int(0.9*total_len) :]
valid_label = labels[int(0.9*total_len) :]

print(len(valid_token))
print(len(valid_label))


input_Layer = tf.keras.Input(shape=(40))

first_embeding = tf.keras.layers.Embedding(10000,20,input_length=40)(input_Layer)
avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(first_embeding)
first_connected = tf.keras.layers.Dense(24,activation=tf.keras.activations.relu)(avg_pooling)
final_dense = tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid)(first_connected)

model = tf.keras.Model(inputs = [input_Layer],outputs=[final_dense])

model.compile(loss = tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['acc'])
model.summary()

model.load_weights('courseraDeepAI/resources/firstWeightSave/')

import numpy as np
model.fit(np.array(train_token),np.array(train_label),
          batch_size=32,
          epochs=1,
          validation_data=(np.array(valid_token),np.array(valid_label)),)
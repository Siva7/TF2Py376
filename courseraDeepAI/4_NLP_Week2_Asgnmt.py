import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd

input_csv = pd.read_csv("courseraDeepAI/resources/bbc-text.csv")

labels_cat = input_csv["category"].astype('category')
labels = labels_cat.cat.codes.to_numpy()
data = input_csv["text"].to_numpy()
print(input_csv["category"].value_counts())
train_data  = data[:2000]
train_label = labels[:2000]

test_data  = data[2000:]
test_label = labels[2000:]


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

my_tokenizer = Tokenizer(10000,oov_token="<OOV>")
my_tokenizer.fit_on_texts(train_data)

train_seq = my_tokenizer.texts_to_sequences(train_data)
print(max(len(x) for x in train_seq))
train_seq_padded = pad_sequences(train_seq,maxlen=4491,padding='post')

test_seq = my_tokenizer.texts_to_sequences(test_data)
test_seq_padded = pad_sequences(test_seq,maxlen=4491,padding='post',truncating='post')


input = tfk.layers.Input(shape=(4491))
embeding_layer = tfk.layers.Embedding(10000,16,input_length=4491)(input)
first_glb_avg_layer = tfk.layers.GlobalAveragePooling1D()(embeding_layer)
first_dense_layer = tfk.layers.Dense(32,activation='relu')(first_glb_avg_layer)
final_layer = tfk.layers.Dense(5,activation='softmax')(first_dense_layer)

model = tfk.models.Model(inputs=[input],outputs=[final_layer])

model.summary()

model.compile(optimizer=tfk.optimizers.Adam(),loss=tfk.losses.sparse_categorical_crossentropy,metrics='acc')

model.fit(train_seq_padded,train_label,epochs=20,validation_data=(test_seq_padded,test_label))
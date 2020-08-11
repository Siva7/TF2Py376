import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist_data  = tf.keras.datasets.fashion_mnist
(X_train,Y_train),(X_test,Y_test) = fashion_mnist_data.load_data()

X_train,Y_train =X_train/255,Y_train
X_test,Y_test = X_test/255,Y_test
X_train.shape
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28],dtype=tf.float32))
model.add(tf.keras.layers.Dense(units=30,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=30,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.keras.activations.softmax))

opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy],
              loss=tf.keras.losses.sparse_categorical_crossentropy)

model.summary()
training_history=model.fit(x=X_train,y=Y_train,epochs=50,verbose=True,validation_split=0.1,
                           callbacks=tf.keras.callbacks.EarlyStopping(patience=4,monitor='sparse_categorical_accuracy',min_delta=0.01,restore_best_weights=True),)

print(training_history.history)

df = pd.DataFrame(training_history.history)

df.plot()
plt.show()

output=model.evaluate(X_test,Y_test)
print(type(output))

print(output)

model.layers

dense_layer_2_weights,dense_layer_2_biases=model.get_layer('dense_1').get_weights()

dense_layer_2_biases.shape
dense_layer_2_weights.shape

for layer in model.layers:
    print(layer.name)

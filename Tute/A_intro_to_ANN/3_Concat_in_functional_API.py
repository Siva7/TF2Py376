import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as tfk

X,Y = fetch_california_housing(return_X_y=True)

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2)

input_layer = tfk.layers.Input(shape=(8))
first_dense_layer = tfk.layers.Dense(10,activation=tfk.activations.relu)(input_layer)
second_dense_layer = tfk.layers.Dense(10,activation=tfk.activations.relu)(first_dense_layer)
concat = tfk.layers.concatenate([input_layer,second_dense_layer],axis=1)
output_layer = tfk.layers.Dense(1)(concat)

model = tfk.Model(inputs=[input_layer],outputs=[output_layer])
model.summary()
opt = tfk.optimizers.Adam(0.01)
model.compile(optimizer=opt,loss=[tfk.losses.mean_squared_error],metrics=[tfk.metrics.mae])

training_detail = model.fit(x=X_train,y=Y_train,
                            batch_size=32,validation_split=0.1,epochs=100,
                            callbacks=tfk.callbacks.EarlyStopping(monitor='loss',min_delta=0.01,patience=10))

model.evaluate(X_test,Y_test)
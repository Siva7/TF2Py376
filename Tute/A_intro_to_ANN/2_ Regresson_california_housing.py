from scipy.stats import zscore
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
def standardize(x):
    scalar = StandardScaler()
    return scalar.fit_transform(x)

X_data,Y_data = fetch_california_housing(return_X_y=True)

X_df=pd.DataFrame(X_data)
Y_df=pd.DataFrame(Y_data)

X_df.shape
Y_df.shape

print(X_df.describe().T.to_string())
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled,columns=X_df.columns)

X_train,X_valid,Y_train,Y_valid  = train_test_split(X_scaled_df,Y_df)

X_train.shape,X_valid.shape,Y_train.shape,Y_valid.shape

input_layer = tf.keras.layers.Input(shape=(8))
second_layer_dense = tf.keras.layers.Dense(10,activation=tf.keras.activations.relu)(input_layer)
final_output = tf.keras.layers.Dense(1)(second_layer_dense)



model = tf.keras.Model(inputs=[input_layer],outputs=[final_output])
def r2_keras(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )

model.compile(loss=tf.keras.losses.mean_squared_error,metrics=[r2_keras],
              optimizer=tf.keras.optimizers.Adam(0.01))

training_hist=model.fit(x=X_train,y=Y_train,validation_split=0.1,
                        callbacks=tf.keras.callbacks.EarlyStopping(monitor='r2_keras',min_delta=0.01,patience=20,
                                                                    verbose=True,restore_best_weights=False),
                        epochs=100)

training_hist_df = pd.DataFrame(training_hist.history)

training_hist_df.plot()



plt.show()

model.evaluate(X_valid,Y_valid)
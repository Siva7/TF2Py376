import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
m_true = 5
c_true = 3
x = np.linspace(0,3,100)
y = m_true * x + c_true + np.random.randn(*x.shape)* 0.5

model = tf.keras.models.Sequential()
single_layer = tf.keras.layers.Dense(units=1,activation=tf.keras.activations.linear,input_shape=(1,))
model.add(single_layer)
sgd_opt = tf.keras.optimizers.SGD(0.01)
loss_fun = tf.keras.losses.mse
metrics_cal = [tf.metrics.Accuracy(),tf.metrics.mse]
model.compile(optimizer=sgd_opt,loss=loss_fun,metrics=metrics_cal)

log_dir = r'./logs'
tensor_flow_call_back = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)


train_history= model.fit(x=x,y=y,epochs=100,validation_split=0.1,verbose=True,callbacks=tensor_flow_call_back)
df=pd.DataFrame(train_history.history)
# print(df.to_string())
df.plot()
plt.show()
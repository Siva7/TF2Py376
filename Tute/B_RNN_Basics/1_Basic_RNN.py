import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk
import datetime
def generate_time_series(batch_size,n_steps):
    freq1 = np.random.rand(batch_size,1)
    time = np.linspace(0,1,n_steps)
    series = 0.2*np.sin((time-freq1)*10)
    noise = 0.1*(np.random.rand(batch_size,n_steps)-0.5)
    print(series.shape)
    print(noise.shape)
    res_series = series+noise
    return  res_series[...,np.newaxis]

n_steps = 50
total_instances = 10000
series = generate_time_series(total_instances,n_steps+1)

print("Series Generated of Shape ::"+str(series.shape))
x_train,y_train = series[:7000,:n_steps],series[:7000,-1]
x_valid,y_valid = series[7000:9000,:n_steps],series[7000:9000,-1]
x_test,y_test = series[9000:,:n_steps],series[9000:,-1]


print("Training Data shape = "+str(x_train.shape)+":"+str(y_train.shape))
print("Training Data shape = "+str(x_valid.shape)+":"+str(y_valid.shape))
print("Training Data shape = "+str(x_test.shape)+":"+str(y_test.shape))
exit()
# Linear Model

input = tfk.layers.Input(shape=(50))
output =  tfk.layers.Dense(1)(input)
model = tfk.Model(inputs=[input],outputs=[output])
model.compile(loss = [tfk.losses.MSE],optimizer = tfk.optimizers.SGD(0.01))

x_train_sqzed = np.squeeze(x_train,axis=-1)
y_train_sqzed = np.squeeze(y_train,axis=-1)

x_valid_sqzed = np.squeeze(x_valid,axis=-1)
y_valid_sqzed = np.squeeze(y_valid,axis=-1)

x_test_sqzed = np.squeeze(x_test,axis=-1)
y_test_sqzed = np.squeeze(y_test,axis=-1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_flow_call_back = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)

model.fit(x=x_train_sqzed,y=y_train_sqzed,epochs=20,validation_data=(x_valid_sqzed,y_valid_sqzed),
          callbacks=[
            tensor_flow_call_back
])

result=model.evaluate(x_test_sqzed,y_test_sqzed)
print("Lineage Regression Base Linea Loss  for 20 epochs"+str(result))

# 0.0026

y_pred = model.predict(x_test_sqzed)

plt.plot(y_pred)
plt.plot(y_test_sqzed)
plt.show()

input_rnn = tfk.layers.Input(shape=(50,1))
simple_rnn = tfk.layers.SimpleRNN(1,activation=tfk.activations.linear)(input_rnn)

rnn_model = tfk.Model(inputs=[input_rnn],outputs=[simple_rnn])
rnn_model.compile(optimizer=tfk.optimizers.Adam(0.01),loss=[tfk.losses.MSE])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_flow_call_back = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)

rnn_model.fit(x=x_train,y=y_train,epochs=20,validation_data=(x_valid,y_valid),
          callbacks=[tensor_flow_call_back])

result=rnn_model.evaluate(x_test,y_test)
print("Basic RNN for 20 epochs "+str(result))
#0.0025


y_pred = rnn_model.predict(x_test)
plt.plot(y_pred)
plt.plot(y_test)
plt.show()


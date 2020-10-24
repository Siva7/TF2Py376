import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tfk
import datetime
import pytz

def get_time_series(batch_size,n_steps):
    freq1 = np.random.rand(batch_size,1)
    time = np.linspace(0,1,n_steps)
    series = 0.2*np.sin((time-freq1)*10)
    noise = 0.1*(np.random.rand(batch_size,n_steps)-0.5)
    print(series.shape)
    print(noise.shape)
    res_series = series+noise
    return  res_series[...,np.newaxis]

n_steps = 50
total_generated_records = 10000
series = get_time_series(total_generated_records,n_steps+1)

print("Generated Series of Shape => "+str(series.shape))
x_train,y_train = series[:7000,:n_steps,:],series[:7000,-1,:]
x_valid,y_valid = series[7000:9000,:n_steps,:],series[7000:9000,-1,:]
x_test,y_test= series[9000:,:n_steps,:],series[9000:,-1,:]

print("Training data shapes => "+str(x_train.shape)+" : "+str(y_train.shape))
print("Validation data shapes => "+str(x_valid.shape)+" : "+str(y_valid.shape))
print("Test data shapes => "+str(x_test.shape)+" : "+str(y_test.shape))

simple_deep_rnn_input = tfk.layers.Input(shape=(None,1),name = "Simple_Deep_RNN_Input")
deep_rnn_layer_one = tfk.layers.SimpleRNN(20,return_sequences=True,name="Deep_RNN_layer_one")(simple_deep_rnn_input)
deep_rnn_layer_two = tfk.layers.SimpleRNN(20,name="Deep_RNN_layer_two")(deep_rnn_layer_one)
final_dense_layer = tfk.layers.Dense(1,name="Final_Dense_Layer")(deep_rnn_layer_two)
# final_dense_layer = tfk.layers.SimpleRNN(1,name="Final_Dense_Layer")(deep_rnn_layer_two)

simple_deep_rnn_model = tfk.models.Model(inputs=[simple_deep_rnn_input],outputs=[final_dense_layer])

simple_deep_rnn_model.compile(loss=[tfk.losses.MSE],optimizer=tfk.optimizers.Adam(0.01),metrics=[tfk.metrics.Accuracy()])

IST = pytz.timezone('Asia/Kolkata')
log_dir = "logs/fit/simple_deep_rnn_" + datetime.datetime.now(IST).strftime("%Y_%m_%d_%H_%M_%S")
tensorboard_calback = tfk.callbacks.TensorBoard(log_dir,histogram_freq=1)
training_hist=simple_deep_rnn_model.fit(x=x_train,y=y_train,epochs=20,callbacks=[tensorboard_calback],validation_data=(x_valid,y_valid))

result = simple_deep_rnn_model.evaluate(x_test,y_test)
print("The loss with  simple_deep_rnn_model for 20 epochs = "+str(result))
print(training_hist.history)

y_pred = simple_deep_rnn_model.predict(x_test)
plt.plot(y_pred.squeeze(axis=-1))
plt.plot(y_test.squeeze(axis=-1))
plt.show()

#The loss with  simple_deep_rnn_model for 20 epochs = [0.000990409404039383, 0.0]
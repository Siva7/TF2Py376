import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
IST = pytz.timezone( 'Asia/Kolkata')

# Have the same time series but at the end instead of predicting just the next step predict next 10 steps

def generate_time_series(batch_size,n_steps):
    time = np.linspace(0,1,n_steps)
    freq = np.random.rand(batch_size,1)
    sine_wave = 0.2*np.sin((time-freq)*10)
    sine_wave_rwo = 0.3 * np.sin((time - freq) * 100)
    noise = 0.1*(np.random.rand(batch_size,n_steps)-0.5)
    series = sine_wave+noise+sine_wave_rwo
    return series[...,np.newaxis]

def plot_predict_vs_actual(x,y_actual,y_preid):
    randInt = np.random.randint(0,x.shape[1])
    x=x[randInt]
    y_actual=y_actual[randInt]
    y_preid=y_preid[randInt]
    actual = np.concatenate([np.squeeze(x),np.squeeze(y_actual)])
    predict = np.concatenate([np.squeeze(x),np.squeeze(y_preid)])
    plt.plot(actual)
    plt.plot(predict)
    plt.title("Entry No :: " + str(randInt))
    plt.show()


total_samples = 10000
step_size=50
series = generate_time_series(10000,step_size+10)

x_train,y_train = series[:7000,:step_size,:],series[:7000,-10:,:]
x_valid,y_valid = series[7000:9000,:step_size,:],series[7000:9000,-10:,:]
x_test,y_test = series[9000:,:step_size,:],series[9000:,-10:,:]

input = tfk.layers.Input(shape=(None,1),name="Inputlayer")
first_rnn_layer = tfk.layers.SimpleRNN(20,return_sequences=True,name="First_RNN_Layer")(input)
second_rnn_layer = tfk.layers.SimpleRNN(20,name="Second_RNN_Layer")(first_rnn_layer)
final_layer = tfk.layers.Dense(10,name="Final_Dense_Layer")(second_rnn_layer)

simple_rnn_model_pred_next_10 = tfk.models.Model(inputs=[input],outputs=[final_layer])

simple_rnn_model_pred_next_10.compile(optimizer=tfk.optimizers.Adam(0.01),loss=[tfk.losses.MSE])

tensorboard_log_dir = "logs/fit/simple_deep_rnn_pred_plus_10_" + datetime.datetime.now(IST).strftime("%Y_%m_%d_%H_%M_%S")

tensorboard_callback = tfk.callbacks.TensorBoard(tensorboard_log_dir,histogram_freq=1)
simple_rnn_model_pred_next_10.fit(x_train,y_train,epochs=20,validation_data=(x_valid,y_valid),callbacks=tensorboard_callback)

result = simple_rnn_model_pred_next_10.evaluate(x_test,y_test)

predict_values = simple_rnn_model_pred_next_10.predict(x_test)
print("Loss for predicitng next 10 values in time => "+str(result))
# Loss for predicitng next 10 values in time => 0.0012660793727263808


plot_predict_vs_actual(x_test,y_test,predict_values)


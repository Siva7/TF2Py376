import numpy as np
import tensorflow.keras as tfk
import datetime
import pytz
import matplotlib.pyplot as plt
import tensorflow as tf
IST = pytz.timezone( 'Asia/Kolkata')

# Train a simple RNN to predict next one step and the
# use that to continiously predect the next 10 results by feeding in the output of one to next one

def get_time_series(batch_size,n_steps):
    time = np.linspace(0,1,n_steps)
    freq = np.random.rand(batch_size,1)
    sine_wave = 0.2*np.sin((time-freq)*10)
    noise = 0.1*(np.random.rand(batch_size,n_steps)-0.5)
    seies = sine_wave+noise
    return seies[...,np.newaxis]

total_data_samples=10000
n_steps = 50
series = get_time_series(total_data_samples,n_steps+10)
x_train,y_train = series[:7000,:n_steps,:],series[:7000,-10:,:]
x_valid,y_valid = series[7000:9000,:n_steps,:],series[7000:9000,-10:,:]
x_test,y_test = series[9000:,:n_steps,:],series[9000:,-10:,:]


simple_deep_rnn_input = tfk.layers.Input(shape=(None,1),name="Simple_deep_rnn_input")
first_layer_deep_rnn = tfk.layers.SimpleRNN(20,name="Simple_deep_rnn_layer_one",return_sequences=True)(simple_deep_rnn_input)
second_layer_deep_rnn = tfk.layers.SimpleRNN(1,return_sequences=False,name="Simple_deep_rnn_layer_two")(first_layer_deep_rnn)
final_layer = tfk.layers.Dense(1,name="Simple_Deep_RNN_final_layer")(second_layer_deep_rnn)

simple_deep_rnn_model = tfk.models.Model(inputs=[simple_deep_rnn_input],outputs=[final_layer])

simple_deep_rnn_model.compile(loss=[tfk.losses.MSE],optimizer=tfk.optimizers.Adam(0.01))

log_dir = "logs/fit/simple_deep_rnn_" + datetime.datetime.now(IST).strftime("%Y_%m_%d_%H_%M_%S")
tensorboard_calback_simple_rnn = tfk.callbacks.TensorBoard(log_dir,histogram_freq=1)


simple_deep_rnn_model.fit(x=x_train,y=y_train[:,0,:],epochs=10,validation_data=(x_valid,y_valid[:,0,:]),callbacks=[tensorboard_calback_simple_rnn])

result = simple_deep_rnn_model.evaluate(x_test,y_test[:,0,:])
print("Simple RNN for 10 epochs has loss of => "+str(result))
#  0.0009912200039252639
y_pred = simple_deep_rnn_model.predict(x_test)
plt.plot(y_pred)
plt.plot(y_test[:,0,:])
plt.show()

# Now using the model to predict next 10 time intervals
x_test_cont =x_test.copy()
for i in range(10):
    x_test_slide = x_test_cont[:,i:i+50,:]
    y_pred_new = simple_deep_rnn_model.predict(x_test_slide)
    x_test_cont = np.concatenate([x_test_cont,y_pred_new[...,np.newaxis]],axis=1)

x_original_cont = series[9000:,:,:]
result_mse=tf.math.reduce_mean(tfk.metrics.MSE(x_original_cont[:,-10:,:],x_test_cont[:,-10:,:]))

print(result_mse)
# tf.Tensor(0.0011847092134449231, shape=(), dtype=float64)


result_mse_last_element=[]
for i in range(10):
    result_mse_last_element.append(tf.math.reduce_mean(tfk.metrics.MSE(x_original_cont[:,-1*i,:],x_test_cont[:,-1*i,:])).numpy())

plt.plot(result_mse_last_element)
plt.show()
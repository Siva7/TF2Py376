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


total_samples = 10000
step_size=50
series = generate_time_series(10000,step_size+10)

x_train,y_train = series[:7000,:step_size,:],series[:7000,-10:,:]
x_valid,y_valid = series[7000:9000,:step_size,:],series[7000:9000,-10:,:]
x_test,y_test = series[9000:,:step_size,:],series[9000:,-10:,:]
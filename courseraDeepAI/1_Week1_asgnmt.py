import tensorflow as tf
import numpy as np
import tensorflow.keras as tfk

xs = np.array([1,2,3,4,5,6])
ys = np.array([1,1.5,2,2.5,3,3.5])

model = tfk.Sequential([tfk.layers.Dense(1)])
model.compile(optimizer=tfk.optimizers.SGD(),loss=tfk.losses.MSE)
model.fit(x=xs,y=ys,epochs=500)


model.predict([7.0])

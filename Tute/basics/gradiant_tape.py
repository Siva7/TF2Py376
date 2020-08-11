import tensorflow as tf
import numpy as np
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

m = tf.Variable(np.random.random([3,4]))
c = tf.Variable(np.random.random([3,3]))
x = tf.Variable(np.random.random([4,3]))
k = tf.constant(np.random.random([3,3]))
with tf.GradientTape(persistent=True) as tape:
    y = tf.matmul(m,x)+c+k

y_m,y_c,y_x,y_k=tape.gradient(y,[m,c,x,k])

with tf.GradientTape(persistent=True) as tape:
    tape.watch(k)
    y = tf.matmul(m,x)+c+k

y_m,y_c,y_x,y_k=tape.gradient(y,[m,c,x,k])


layer = tf.keras.layers.Dense(2,activation=tf.keras.activations.relu)
x = tf.constant([[10,20,30]])

with tf.GradientTape(persistent=True) as tape2:
    y = layer(x)
    loss = tf.reduce_sum(y)

loss_grad = tape2.gradient(loss,layer.trainable_variables)
print(loss_grad)
output_grad =tape2.gradient(y,layer.trainable_variables)
print(output_grad)
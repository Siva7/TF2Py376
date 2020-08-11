import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
m_true = 5
c_true = 3
x = np.linspace(0,3,100)
y = m_true * x + c_true + np.random.randn(*x.shape)* 0.5
plt.figure(figsize=(10,8))
plt.scatter(x,y)
plt.show()

m_var = tf.Variable(np.random.randn(),name="m")
c_var = tf.Variable(np.random.randn(),name="m")

def epoch():
    with tf.GradientTape(persistent=True) as tape:
        y_pred = m_var * x + c_var
        loss =tf.reduce_mean(tf.square(y_pred-y))
        grad_loss_m_avar ,grad_loss_c_avar = tape.gradient(loss,[m_var,c_var])
        m_var.assign_sub(0.01*grad_loss_m_avar)
        c_var.assign_sub(0.01 * grad_loss_c_avar)
        return loss,m_var,c_var

for i in range(100):
    loss,m,c = epoch()
    print("Epoc "+str(i)+" loss="+str(loss.numpy())+" m="+str(m.numpy())+ " c="+str(c.numpy()))



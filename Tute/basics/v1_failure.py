import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
c= tf.add(a,b)
print(c)

sess = tf.compat.v1.Session()
sess.run(fetches=c)


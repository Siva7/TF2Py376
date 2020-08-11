import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a,b)
print(c)

with tf.compat.v1.Session() as sess:
    print(sess.run(c))
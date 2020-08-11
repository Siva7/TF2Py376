import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

m = tf.Variable([20],shape=[1])
c = tf.Variable([30],shape=[1])
x = tf.compat.v1.placeholder(shape=[1],dtype=tf.int32)
y = m*x+c


init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    out = sess.run(fetches=y,feed_dict={x:[200]})
    print(out)
    writer = tf.compat.v1.summary.FileWriter('./logs',sess.graph)
    writer.close()
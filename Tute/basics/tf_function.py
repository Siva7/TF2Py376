import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
@tf.function
def add(a,b):
    for i in tf.range(a):
        c=b+a
        b=c
    return b
x = tf.Variable(3)
y = tf.Variable(2)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
z = add(x,y)
print(sess.run(z))
writer = tf.compat.v1.summary.FileWriter('./logs',sess.graph)
writer.close()
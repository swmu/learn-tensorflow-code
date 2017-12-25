import tensorflow as tf

a = tf.constant([3,3], name="a")
b = tf.constant([[0,1],[2,3]], name="b")

x = tf.add(a,b, name="add")
y = tf.multiply(a,b, name="mul")

with tf.Session() as sess:
    write = tf.summary.FileWriter("./graphs", sess.graph)
    x, y = sess.run([x, y])
    print(x, y)

write.close()
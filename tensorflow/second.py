import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")



with tf.Session() as sess:
    # add this line to use TensorBoard.
    writer = tf.summary.FileWriter("graphs", sess.graph)
    c = sess.run(x)
    print(c)

writer.close() # close the writer when you’re done using it
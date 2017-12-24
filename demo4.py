import tensorflow as tf

# W = tf.Variable(tf.truncated_normal([7, 3]))
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print(W.eval())

my_var = tf.Variable(2, name="my_var")
# assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
    print(sess.run(my_var.initializer))
    print(sess.run(my_var_times_two)) # >> 4
    print(sess.run(my_var)) # >> ?????
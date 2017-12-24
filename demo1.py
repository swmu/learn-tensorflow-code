import tensorflow as tf

t_2 = [
    [True, False, False],
    [False, False, True],
    [False, True, False]
]

zeros = tf.zeros_like(t_2)
ones = tf.ones_like(t_2)

with tf.Session() as sess:
    z = sess.run(ones)
    print(z)
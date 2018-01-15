# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main(data_dir):
    # 获取数据
    pass
    # one_hot=True 表示将数据换位一个只有一个位置为1的向量 如 [0,0,0,0,0,1,0,0,0]
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    #  y = wx +b
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, w) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    # loss函数, 获取结果为[n_class]的向量
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    # 求均值
    # x = [[1,2],[3,4]]
    # tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
    # tf.reduce_mean(x, 0) ==> [2.,  3.] #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
    # tf.reduce_mean(x, 1) ==> [1.5,  3.5] #指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
    cross_entrop = tf.reduce_mean(loss)
    # 设置loss函数最优化
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entrop)

    # 如上部分是构建了一个图， 如下部分是需要把图进行运行以便达到训练的目的

    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()
        # 开始训练
        # 表示循环训练1000次
        for i in range(1000):
          # 每一个批次100条数据
          batch_xs, batch_ys = mnist.train.next_batch(100)
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  data_dir = '/tmp/tensorflow/mnist/input_data'
  main(data_dir)
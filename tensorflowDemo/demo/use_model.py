import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
  data_dir = '/tmp/tensorflow/mnist/input_data'
  # one_hot=True 表示将数据换位一个只有一个位置为1的向量 如 [0,0,0,0,0,1,0,0,0]
  mnist = input_data.read_data_sets(data_dir, one_hot=True)
  saver = tf.train.import_meta_graph("./model/minist.ckpt.meta")
  with tf.Session() as sess:
     saver.restore(sess, "./model/minist.ckpt")
     graph = tf.get_default_graph()
     x = graph.get_operation_by_name('x').outputs[0]
     y_ = graph.get_operation_by_name('y').outputs[0]
     accuracy = graph.get_operation_by_name('accuracy').outputs[0]
     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
     #  在使用t.eval()时，等价于：tf.get_default_session().run(t)
     print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



if __name__ == '__main__':
    main()
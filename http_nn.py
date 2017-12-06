
# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf
import numpy

word_size = 100
max_words = 128

# NN variables, same for everyone
W = tf.Variable(tf.random_normal([6656,5]), trainable=True)
b = tf.Variable(tf.random_normal([5]), trainable=True)

filter_1st_layer = tf.Variable(tf.random_normal([5,5,1,1]), trainable=True)
bias_1st =         tf.Variable(tf.zeros([max_words]))

filter_2nd_layer = tf.Variable(tf.random_normal([3,3,64,64]), trainable=True)
bias_2nd =         tf.Variable(tf.zeros([max_words/2]))

def get_nn(input, labels, mode):
    #keep_prob=0.8
    input= tf.reshape(input, [-1, 128, 100, 1])
    print input

    one = tf.layers.conv2d (inputs=input,
                            filters=64,
                            kernel_size=[ 32, 4 ],
                            padding="same",
                            activation=tf.nn.relu)
    one_mp = tf.layers.max_pooling2d(inputs=one, pool_size=[2,2],strides=2)
    print one_mp
    two = tf.layers.conv2d (inputs=one_mp,
                            filters=128,
                            kernel_size=[32,4 ],
                            padding="same",
                            activation=tf.nn.relu)
    two_mp = tf.layers.max_pooling2d(inputs=two, pool_size=[2,2], strides=2)
    print two_mp
    # Dense Layer
    pool2_flat = tf.reshape(two_mp, [-1, 32*25*128])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    print dropout

    # Logits Layer
    y__ = tf.layers.dense(inputs=dropout, units=5)
    y = tf.nn.softmax(y__)
    
    print y
    
    #y             = tf.nn.softmax(tf.add(tf.matmul(resh, W), b)) #y = tf.nn.relu(tf.matmul(x,W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return y, cross_entropy, accuracy

def parse_record ( filename, batch_s, shuffle ):
    filename_queue = tf.train.string_input_producer([filename])
    reader     = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features   = tf.parse_single_example(
        value,
        features={
            'train/label': tf.FixedLenFeature([5],       tf.int64),
            'train/words': tf.FixedLenFeature([word_size * max_words], tf.float32)
        }
    )
    label = tf.cast(features['train/label'], tf.float32)
    line  = tf.reshape(tf.cast(features['train/words'], tf.float32), [max_words, word_size])
    if shuffle:
        return tf.train.shuffle_batch([label, line], batch_size=batch_s, capacity=4, min_after_dequeue=1)
    else:
        return tf.train.batch([label, line], batch_size=batch_s, capacity=10 )


filename_queue_test = tf.train.string_input_producer(["simple_real.tfrecords2"])

label,      line      = parse_record("http_train.tfrec", 10,  False)
label_test, line_test = parse_record("http_valid.tfrec", 10, True)

y, ce, acc       = get_nn(line,      label,      tf.estimator.ModeKeys.TRAIN)
y_t, ce_t, acc_t = get_nn(line_test, label_test, tf.estimator.ModeKeys.EVAL)

train_step = tf.train.AdamOptimizer(0.003).minimize(ce)
model = tf.global_variables_initializer()
coord = tf.train.Coordinator()
NUM_THREADS=2
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as session:
    #saver.restore(session, "/tmp/model.ckpt")
    #print("Model restored.")
    #print W.eval()
    #sys.exit(0)
    session.run(model)
    threads=tf.train.start_queue_runners(sess=session, coord=coord)
    
    for ii in range(1500):
        print ".",
        sys.stdout.flush()
        ttt = session.run(train_step) # , feed_dict={xx: line, y_: label})
        if (ii%10 == 9):
            print "\nEvaluating..."
            values= session.run([W,acc_t, y_t, label_test, line_test]) # , feed_dict={xx: line_test.eval(), y_: label_test.eval() })
            #print str(values[0])
            print "output:" +str(values[2])
            print "labels: "+str(values[3])
            print "accuracy: "+ str(values[1])
            print str(ii)
        

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(session, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

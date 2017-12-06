
# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf
import numpy

# NN variables, same for everyone
W = tf.Variable(tf.random_normal([24,2]), trainable=True)
b = tf.Variable(tf.random_normal([2]), trainable=True)

def get_nn(input, labels):
    #keep_prob=0.8
    
    y             = tf.nn.softmax(tf.add(tf.matmul(input ,W), b)) #y = tf.nn.relu(tf.matmul(x,W) + b)
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
            'train/label': tf.FixedLenFeature([2],  tf.int64),
            'train/line':  tf.FixedLenFeature([24], tf.int64)
        }
    )
    label = tf.cast(features['train/label'], tf.float32)
    line  = tf.cast(features['train/line'],  tf.float32)
    if shuffle:
        return tf.train.shuffle_batch([label, line], batch_size=batch_s, capacity=4, min_after_dequeue=1)
    else:
        return tf.train.batch([label, line], batch_size=batch_s, capacity=10 )


filename_queue_test = tf.train.string_input_producer(["simple_real.tfrecords2"])

label,      line      = parse_record("simple.tfrecords2",      24, False)
label_test, line_test = parse_record("simple_real.tfrecords2", 100,  True)

y, ce, acc       = get_nn(line,      label)
y_t, ce_t, acc_t = get_nn(line_test, label_test)

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
        
        ttt = session.run(train_step) # , feed_dict={xx: line, y_: label})
        if (ii%100 == 99):
            values= session.run([W,acc_t, y_t, label_test, line_test]) # , feed_dict={xx: line_test.eval(), y_: label_test.eval() })
            print str(values[0])
            #print "output:" +str(values[2])
            #print "labels: "+str(values[3])
            print "accuracy: "+ str(values[1])
            print str(ii)
        

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(session, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)


# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

from tensorflow.examples.tutorials.mnist import input_data
import sys
import tensorflow as tf
import numpy

word_size = 100
max_words = 128

tf.logging.set_verbosity(tf.logging.INFO)

def get_nn(features, labels, mode):
    #keep_prob=0.8
    input= tf.reshape(features["train/words"], [-1, 128, 100, 1])
    
    one = tf.layers.conv2d (inputs=input,
                            filters=32,
                            kernel_size=[ 32, 4 ],
                            padding="same",
                            activation=tf.nn.relu)
    one_mp = tf.layers.max_pooling2d(inputs=one, pool_size=[2,2],strides=2)
    two = tf.layers.conv2d (inputs=one_mp,
                            filters=64,
                            kernel_size=[32,4 ],
                            padding="same",
                            activation=tf.nn.relu)
    two_mp = tf.layers.max_pooling2d(inputs=two, pool_size=[2,2], strides=2)
    # Dense Layer
    pool2_flat = tf.reshape(two_mp, [-1, 32*25*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Logits Layer
    y__ = tf.layers.dense(inputs=dropout, units=5)
    y = tf.nn.softmax(y__)
    
    print y
    predictions = tf.reshape(y, [-1])
    loss = tf.losses.mean_squared_error(labels, predictions)

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float32), predictions),
        "labels": labels,
        "output": predictions
    }
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    predictions_dict = {
        "result": predictions,
        }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        predictions=predictions)
    #return y, cross_entropy, accuracy

def train_fn():
    return parse_record("http_train.tfrec", 100)

def eval_fn():
    return parse_record("http_valid.tfrec", 10)
    
def parse_record ( filename, batch_s ):
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
    #return features
    label = tf.cast(features['train/label'], tf.float64)
    line  = tf.reshape(tf.cast(features['train/words'], tf.float64), [max_words, word_size])
    return features, label

nn = tf.estimator.Estimator(model_fn=get_nn)

for i in range(1,1):
    nn.train(input_fn=train_fn, steps=100)

print "Now evaluating"

ev = nn.evaluate(input_fn=eval_fn, steps=10)
print("Loss: %s" % ev["loss"])
print("Root Mean Squared Error: %s" % ev["rmse"])
print("labels"+ ev["labels"]);
print("preds"+  ev["output"]);



# label,      line      = 
# label_test, line_test = 



#y, ce, acc       = get_nn(line,      label,      tf.estimator.ModeKeys.TRAIN)
#y_t, ce_t, acc_t = get_nn(line_test, label_test, tf.estimator.ModeKeys.EVAL)







# train_step = tf.train.AdamOptimizer(0.003).minimize(ce)
# model = tf.global_variables_initializer()
# coord = tf.train.Coordinator()
# NUM_THREADS=2
# saver = tf.train.Saver()

# with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as session:
#     #saver.restore(session, "/tmp/model.ckpt")
#     #print("Model restored.")
#     #print W.eval()
#     #sys.exit(0)
#     session.run(model)

    
#     for ii in range(1500):
#         print ".",
#         sys.stdout.flush()
#         ttt = session.run(train_step) # , feed_dict={xx: line, y_: label})
#         if (ii%10 == 9):
#             print "\nEvaluating..."
#             values= session.run([W,acc_t, y_t, label_test, line_test]) # , feed_dict={xx: line_test.eval(), y_: label_test.eval() })
#             #print str(values[0])
#             print "output:" +str(values[2])
#             print "labels: "+str(values[3])
#             print "accuracy: "+ str(values[1])
#             print str(ii)
        

#     coord.request_stop()
#     coord.join(threads)

#     save_path = saver.save(session, "/tmp/model.ckpt")
#     print("Model saved in file: %s" % save_path)

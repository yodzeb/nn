
import fileinput
import tensorflow as tf
import re
import sys

train_filename="./simple_real.tfrecords2"

writer = tf.python_io.TFRecordWriter(train_filename)

m=re.compile(r".*\d\s[12]\d:.*") # Late

dataset=[]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for i in range(1,100000):
    line=i%24

    #for line in fileinput.input():
    #line=line[11:13]
    #print '0'+str(line)+'9'
    val = int(line)
    line = [0]*24
    line[val] = 1
    
    label= [1, 0]  # early
    # if re.match(m, line) is not None:
    if val > 12:
        label= [0,1]
    #line2 = [ord (x) for x in line]
    record={"li" :line,
            "la" :label}
    #print str(label)+line
    dataset.append(record)

    
for i in range(len(dataset)):
    feature = {'train/label': _int64_list_feature(dataset[i]["la"]),
               'train/line':  _int64_list_feature(dataset[i]["li"])}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
    

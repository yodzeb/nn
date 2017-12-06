
from optparse import OptionParser
import gzip
import tensorflow as tf
import re
import sys
import gensim
import numpy as np

sentence_size=128

filename = "/home/yodzeb/Downloads/http.log.gz"

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list_feature (value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def classify (line):
    # Classification
    # [ SQL, LFI, BIN, WP, OTHER ]
    labels=[0,0,0,0]
    if re.search(r'select', line) and re.search(r'from', line):
        labels = [1,0,0,0,0]
    elif re.search(r'\.(?:\/|\\)\.', line) or re.search(r'(?:passwd|boot\.ini|winnt)', line):
        labels = [0,1,0,0,0]
    elif re.search(r'\\x..\\x', line):
        labels = [0,0,1,0,0]
    elif re.search(r'wp-', line):
        labels = [0,0,0,1,0]
    else:
        labels = [0,0,0,0,1]
    return labels


def do_file (filename, c, model, vec_size, out_file):
    handle = gzip.open (filename, "r");
    i=0
    writer = tf.python_io.TFRecordWriter(out_file)
    for line in handle:
        i=i+1
        if (i%100 == 0):
            print ".",
            sys.stdout.flush()
        if i==c:
            break
        req=line.split("\t")[9]
        words = re.split(r'(?:\/|=|\?|\\|\&|\s|\'|,|\+)', req)
        words = map(str.lower, words)

        output = []
        count = 0
        for w in words:
            count=count+1
            if w in model.wv.vocab:
                output.append(model[w].tolist())
            else:
                output.append([0]*vec_size)
            if count == sentence_size:
                break
        while count < sentence_size:
            output.append([0]*vec_size)
            count=count+1
                
        labels = classify(line)
        record = { "la": labels, "wo": np.reshape(output, -1)   }
        #print record
        write_record(writer, record)
        
def write_record ( out, record ):
    feature = {'train/label': _int64_list_feature(record["la"]),
               'train/words': _float_list_feature(record["wo"])}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    out.write(example.SerializeToString())

# output files
train_filename="./http_train.tfrec";
valid_filename="./http_valid.tfrec";

parser = OptionParser()
parser.add_option("-f", "--file", action="store", type="string", dest="filename")
(options, args) = parser.parse_args()
if (options.filename):
    filename=options.filename

# Load dictionary
model = gensim.models.Word2Vec.load('./mymodel-300k')
vec_size = len(model['index.php'])
print "Vectors size: "+str(vec_size)

do_file("/home/yodzeb/Downloads/http_train.gz", 100000, model, vec_size, train_filename)
do_file("/home/yodzeb/Downloads/http_valid.gz", 15000,  model, vec_size, valid_filename)

#write_dataset(data_train, train_filename)
#write_dataset(data_valid, valid_filename)
    
